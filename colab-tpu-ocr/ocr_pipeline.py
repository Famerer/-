# %% [markdown]
# # OCR на TPU v5e-1: буквы, цифры, слова, пробелы, многострочность, разные поверхности
#
# Самодостаточный ноутбук: ничего не скачивается, обучающие данные генерируются здесь же.
#
# Что умеет:
# - **латиница A-Z / a-z и цифры 0-9** — CNN-классификатор символов;
# - **слова и пробелы** — пробел определяется по геометрии межсимвольных зазоров;
# - **несколько строк** — сегментация строк по проекции чернил, затем чтение каждой;
# - **разные поверхности** — бумага, градиент, блик, доска, камень, дерево + шум,
#   размытие, поворот, яркость/контраст; инференс использует локальный контраст,
#   поэтому одинаково работает и на тёмном, и на светлом фоне.
#
# Конвейер: `адаптивный контраст -> сегментация строк -> сегментация символов
# -> CNN (TPU/GPU/CPU) -> восстановление пробелов по зазорам -> сборка текста`.
#
# Ускоритель выбирается автоматически: TPU (torch_xla) -> GPU (CUDA) -> CPU.
# В Colab: Среда выполнения -> Сменить среду выполнения -> **TPU v5e-1**, затем «Выполнить все».
# Локальная проверка: `OCR_SMOKE=1 python ocr_pipeline.py`.
#
# %% [markdown]
# ## 1. Настройки, ускоритель, алфавит
# %%
import os, sys, math, random, string, time, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SMOKE = os.environ.get("OCR_SMOKE", "0") == "1"
IS_COLAB = "google.colab" in sys.modules
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

XM = None


def detect_device():
    global XM
    try:
        import torch_xla.core.xla_model as xm
        XM = xm
        dev = xm.xla_device()
        return dev, "tpu", "TPU (torch_xla): " + str(dev)
    except Exception as exc:
        note = "torch_xla недоступен (%s)" % type(exc).__name__
    if torch.cuda.is_available():
        return torch.device("cuda"), "gpu", note + " -> CUDA: " + torch.cuda.get_device_name(0)
    return torch.device("cpu"), "cpu", note + " -> CPU"


DEVICE, DEVICE_KIND, DEVICE_INFO = detect_device()
if DEVICE_KIND == "tpu":
    try:
        import torch_xla.runtime as xr
        DEVICE_INFO += " | device_type=" + str(xr.device_type())
    except Exception:
        pass

CFG = dict(
    char_size=32,
    n_train=12400 if SMOKE else 60000,
    epochs=6 if SMOKE else 10,
    batch=128,
    c=32,
    lr=2e-3,
    n_test=120 if SMOKE else 600,
    n_multiline=30 if SMOKE else 160,
)
# Colab иногда выдаёт CPU без ускорителя — урезаем объём, чтобы не считать часами.
if not SMOKE and DEVICE_KIND == "cpu":
    print("Внимание: выдан CPU-рантайм, автоматически уменьшаю объём обучения")
    CFG.update(n_train=16000, epochs=6, batch=128)
if os.environ.get("OCR_QUICK") == "1":
    CFG.update(n_train=2500, epochs=2, n_test=40, n_multiline=10)

OUT = Path(os.environ.get("OCR_OUT", str(Path(__file__).resolve().parent / "out")))
OUT.mkdir(parents=True, exist_ok=True)
print("Ускоритель:", DEVICE_INFO)
print("Режим:", "SMOKE" if SMOKE else "ПОЛНОЕ ОБУЧЕНИЕ", "| конфиг:", json.dumps(CFG))
print("Выходная папка:", OUT)

CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase
CHAR2IDX = {ch: i for i, ch in enumerate(CHARS)}
IDX2CHAR = {i: ch for i, ch in enumerate(CHARS)}
NCLS = len(CHARS)
SURFACES = ["paper", "gradient", "glare", "chalkboard", "stone", "wood"]
SMOKE_WORDS = [
    "cat", "dog", "sun", "moon", "code", "game", "tpu", "ai", "robot", "pixel",
    "level", "boss", "stone", "wood", "paper", "light", "dark", "word", "line", "read",
    "hello", "world", "test", "rust", "godot", "2026", "42", "x9", "abc", "zero",
]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/comic.ttf",
]
FONTS = [p for p in FONT_CANDIDATES if Path(p).is_file()]
print("Классов:", NCLS, "| шрифтов:", len(FONTS))


class R:
    def __init__(self, seed):
        self.p = random.Random(seed)
        self.n = np.random.default_rng(seed % (2 ** 32))

    def randint(self, a, b):
        return self.p.randint(a, b)

    def choice(self, seq):
        return self.p.choice(seq)

    def random(self):
        return self.p.random()

    def uniform(self, a, b):
        return self.p.uniform(a, b)

    def normal(self, mu, sigma, size):
        return self.n.normal(mu, sigma, size).astype(np.float32)

    def randarray(self, h, w):
        return self.n.random((h, w)).astype(np.float32)


def lowfreq(rng, h, w, scale=12):
    sh, sw = max(2, h // scale), max(2, w // scale)
    small = (rng.randarray(sh, sw) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR)).astype(np.float32) - 127.5


def make_surface(rng, w, h, kind):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lf, n = lowfreq(rng, h, w, 12), rng.normal(0, 1, (h, w))
    dark = False
    if kind == "paper":
        base = 236 + lf * 0.10 + n * 3
    elif kind == "gradient":
        base = 228 - 75 * (xx / w) + lf * 0.12 + n * 4
    elif kind == "glare":
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        base = 232 + lf * 0.10 + n * 3 + 85 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.25) ** 2)))
    elif kind == "chalkboard":
        base, dark = 44 + lf * 0.30 + n * 6, True
    elif kind == "stone":
        base = 152 + lf * 0.55 + n * 10
    elif kind == "wood":
        base = 174 + 18 * np.sin(yy * 0.22 + xx * 0.012) + lf * 0.35 + n * 7
    else:
        base = 230 + lf * 0.1 + n * 4
    base = np.clip(base, 0, 255)
    tint = rng.normal(0, 5, 3).astype(np.float32)
    return np.clip(base[..., None] + tint[None, None, :], 0, 255).astype(np.uint8), dark


def pick_font(rng, size=None):
    if not FONTS:
        try:
            return ImageFont.load_default(size=size or 30), size or 30
        except TypeError:
            return ImageFont.load_default(), 12
    size = size or rng.randint(26, 36)
    return ImageFont.truetype(rng.choice(FONTS), size), size


def composite(mask, rgb, dark, rng, h, w):
    a = np.asarray(mask).astype(np.float32) / 255.0
    a *= np.clip(1 + lowfreq(rng, h, w, 7) / 420, 0.72, 1.28)
    a = np.clip(a, 0, 1)
    ink = rng.normal(228 if dark else 26, 7 if dark else 9, 3).astype(np.float32)
    ink = np.clip(ink, 0, 255)
    img = rgb.astype(np.float32) * (1 - a[..., None]) + ink[None, None, :] * a[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def augment(im, rng, rgb, strong=False):
    fill = tuple(int(v) for v in rgb[0, 0])
    if rng.random() < (0.75 if strong else 0.4):
        im = im.rotate(rng.uniform(-4 if strong else -1.6, 4 if strong else 1.6),
                       resample=Image.BILINEAR, fillcolor=fill)
    if rng.random() < 0.45:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    arr = np.asarray(im).astype(np.float32)
    arr = np.clip((arr - 127.5) * rng.uniform(0.8, 1.3) + 127.5 + rng.uniform(-16, 16), 0, 255)
    arr = arr + rng.normal(0, 5, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L")


def draw_text(d, text, font, x, y, size, rng):
    for ch in text:
        if ch == " ":
            x += size * 0.5 + rng.randint(0, 4)
            continue
        sw = 1 if rng.random() < 0.35 else 0
        d.text((x, y), ch, font=font, fill=255, anchor="lt", stroke_width=sw, stroke_fill=255)
        x += font.getlength(ch) + rng.uniform(0.4, 2.0)
    return x


def random_word(rng):
    if SMOKE:
        return rng.choice(SMOKE_WORDS)
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(2, 8)))


def random_line(rng):
    words = [random_word(rng) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.22:
        words = [w.upper() for w in words]
    return " ".join(words)


# %% [markdown]
# ## 2. Синтетические данные: одиночные символы, строки, документы
# %%
def render_char(ch, rng, clean=False):
    S = 40
    rgb, dark = make_surface(rng, S, S, rng.choice(SURFACES))
    font, size = pick_font(rng, rng.randint(24, 34))
    mask = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mask)
    w = font.getlength(ch)
    x = (S - w) / 2 + rng.uniform(-3, 3)
    y = (S - size) / 2 - 2 + rng.uniform(-3, 3)
    d.text((x, y), ch, font=font, fill=255, anchor="lt",
           stroke_width=1 if rng.random() < 0.3 else 0, stroke_fill=255)
    im = composite(mask, rgb, dark, rng, S, S)
    return im.convert("L") if clean else augment(im, rng, rgb, strong=True)


def norm_img(gray, size=32, pad_value=None):
    arr = np.asarray(gray.convert("L"), dtype=np.float32) / 255.0
    z = np.clip((arr - arr.mean()) / (arr.std() + 1e-5), -3, 3) / 3.0
    im = Image.fromarray(np.clip((z + 1) * 127.5, 0, 255).astype(np.uint8))
    bw, bh = im.size
    scale = min(size / bw, size / bh)
    nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    bg = int(pad_value if pad_value is not None else np.median(np.asarray(im)))
    canvas = Image.new("L", (size, size), bg)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return torch.from_numpy(a[None, :, :].copy())


class CharDataset(Dataset):
    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = R(self.seed + i * 7919)
        ch = CHARS[i % NCLS]
        return norm_img(render_char(ch, rng, clean=(rng.random() < 0.45))), CHAR2IDX[ch]


def render_line(text, rng, force_surface=None, w=320):
    h = 48
    rgb, dark = make_surface(rng, w, h, force_surface or rng.choice(SURFACES))
    font, size = pick_font(rng)
    est = lambda f, s: sum((s * 0.5 + 2) if c == " " else (f.getlength(c) + 1.4) for c in text)
    while est(font, size) > w - 24 and size > 16:
        size -= 2
        font, _ = pick_font(rng, size)
    tw = est(font, size)
    x = rng.randint(10, max(11, int(w - tw - 10)))
    y = (h - size) // 2 - 2 + rng.randint(-2, 2)
    mask = Image.new("L", (w, h), 0)
    draw_text(ImageDraw.Draw(mask), text, font, x, y, size, rng)
    return augment(composite(mask, rgb, dark, rng, h, w), rng, rgb)


def render_multiline(lines, rng, force_surface=None):
    w, lh, pad = 320, 64, 20
    h = pad * 2 + len(lines) * lh
    rgb, dark = make_surface(rng, w, h, force_surface or rng.choice(SURFACES))
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for i, text in enumerate(lines):
        font, size = pick_font(rng)
        x, y = rng.randint(10, 34), pad + i * lh + (lh - size) // 2 - 2
        draw_text(d, text, font, x, y, size, rng)
    return augment(composite(mask, rgb, dark, rng, h, w), rng, rgb)


# %% [markdown]
# ## 3. CNN-классификатор символов и обучение
# %%
class CharNet(nn.Module):
    def __init__(self, ncls, c=32):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 4 * 16, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(256, ncls))

    def forward(self, x):
        return self.head(self.feat(x))


model = CharNet(NCLS, CFG["c"]).to(DEVICE)
loader = DataLoader(CharDataset(CFG["n_train"], SEED + 1), batch_size=CFG["batch"],
                    shuffle=True, num_workers=0, drop_last=False)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
print("Параметров модели:", sum(p.numel() for p in model.parameters()))


def save_ckpt(path):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)


best, t0 = float("inf"), time.time()
for ep in range(1, CFG["epochs"] + 1):
    model.train()
    tot, correct, seen = 0.0, 0, 0
    for xs, ys in loader:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        opt.step()
        if DEVICE_KIND == "tpu":
            XM.mark_step()
        tot = tot + loss.detach()
        correct = correct + (logits.argmax(1) == ys).sum().detach()
        seen += ys.numel()
    sched.step()
    l = float(tot.cpu()) / max(1, len(loader)) if torch.is_tensor(tot) else tot / max(1, len(loader))
    acc = float(correct.cpu()) / max(1, seen)
    if l < best:
        best = l
        save_ckpt(OUT / "char_cnn.pt")
    print("эпоха %d/%d  loss=%.4f  symbol_acc=%.3f  t=%.1fs" % (ep, CFG["epochs"], l, acc, time.time() - t0))
print("Обучение заняло %.1f c" % (time.time() - t0))
model.load_state_dict(torch.load(OUT / "char_cnn.pt", map_location="cpu"))
model.to(DEVICE)
model.eval()


# %% [markdown]
# ## 4. Инференс: сегментация строк и символов, восстановление пробелов
# %%
def ink_mask(gray, radius=18, pct=92):
    im = gray.convert("L")
    arr = np.asarray(im).astype(np.float32)
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
    d = np.abs(arr - bg)
    thr = float(np.clip(np.percentile(d, pct), 15, 80))
    return d > thr


def runs_of(active, min_len=1):
    out, start = [], None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append([start, i - 1])
            start = None
    if start is not None and len(active) - start >= min_len:
        out.append([start, len(active) - 1])
    return out


def merge_runs(runs, max_gap):
    out = []
    for a, b in runs:
        if out and a - out[-1][1] <= max_gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def smooth(v, k):
    return np.convolve(v, np.ones(k) / k, mode="same")


@torch.no_grad()
def classify_crops(crops):
    if not crops:
        return []
    x = torch.cat([norm_img(c) for c in crops], 0).unsqueeze(1).to(DEVICE)
    preds = model(x).argmax(1).cpu().numpy().tolist()
    return [IDX2CHAR[p] for p in preds]


def segment_lines(gray):
    m = ink_mask(gray, radius=22)
    proj = smooth(m.sum(axis=1).astype(np.float32), 5)
    active = proj > max(2.0, 0.15 * float(proj.max()))
    groups = merge_runs(runs_of(active, 3), 6)
    out = []
    for a, b in groups:
        if b - a < 10:
            continue
        a, b = max(0, a - 4), min(gray.height - 1, b + 4)
        out.append(gray.crop((0, a, gray.width, b + 1)))
    return out


@torch.no_grad()
def read_line(gray):
    m = ink_mask(gray, radius=14)
    col = smooth(m.sum(axis=0).astype(np.float32), 3)
    active = col > max(2.0, 0.18 * float(col.max()))
    groups = merge_runs(runs_of(active, 2), 2)
    widths = [b - a + 1 for a, b in groups]
    mw = float(np.median(widths)) if widths else 12.0
    # Слишком широкие компоненты (слипшиеся буквы) режем на примерно равные части.
    split = []
    for a, b in groups:
        w = b - a + 1
        if w > 1.75 * mw:
            n = max(2, int(round(w / mw)))
            step = w / n
            split += [[int(a + i * step), int(a + (i + 1) * step) - 1] for i in range(n)]
        else:
            split.append([a, b])
    groups = [g for g in split if g[1] - g[0] >= 1]
    if not groups:
        return ""
    crops = [gray.crop((max(0, a - 2), 0, min(gray.width, b + 3), gray.height)) for a, b in groups]
    chars = classify_crops(crops)
    space_thr = max(5.0, 0.48 * mw)
    text = chars[0]
    for i in range(1, len(groups)):
        gap = groups[i][0] - groups[i - 1][1]
        if gap > space_thr:
            text += " "
        text += chars[i]
    return " ".join(text.split())


@torch.no_grad()
def predict_document(gray):
    lines = segment_lines(gray)
    return "\n".join(read_line(l) for l in lines) if lines else ""


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(s.replace("\n", " ").split())


def score(gt, pred):
    g, p = norm(gt), norm(pred)
    cer = edit_distance(g, p) / max(1, len(g))
    wa = float(g.split() == p.split())
    gs = {i for i, ch in enumerate(g) if ch == " "}
    ps = {i for i, ch in enumerate(p) if ch == " "}
    sf1 = 2 * len(gs & ps) / max(1, len(gs) + len(ps))
    return cer, wa, sf1


def summarize(rows, label):
    cer = float(np.mean([r[3] for r in rows]))
    wa = float(np.mean([r[4] for r in rows]))
    sf = float(np.mean([r[5] for r in rows]))
    print("%-24s CER=%.4f  word_acc=%.3f  space_F1=%.3f  (n=%d)" % (label, cer, wa, sf, len(rows)))
    return dict(cer=cer, word_acc=wa, space_f1=sf, n=len(rows))


# %% [markdown]
# ## 5. Проверка: поверхности, пробелы, многострочность
# %%
test_rows = []
for i in range(CFG["n_test"]):
    rng = R(555001 + i * 104729)
    surf = SURFACES[i % len(SURFACES)]
    text = random_line(rng)
    pred = read_line(render_line(text, rng, force_surface=surf))
    test_rows.append((surf, text, pred, *score(text, pred)))
test_m = summarize(test_rows, "ТЕСТ (все поверхности)")
by_surface = {}
for idx, s in enumerate(SURFACES):
    sub = [r for r in test_rows if r[0] == s]
    if not sub:
        sub = []
        for j in range(30):
            rng = R(7000 + idx * 333 + j)
            txt = random_line(rng)
            sub.append((s, txt, read_line(render_line(txt, rng, force_surface=s)), 0, 0, 0))
        sub = [(s, t, p, *score(t, p)) for s, t, p, *_ in sub]
    by_surface[s] = summarize(sub, "  поверхность: " + s)

multi = []
for i in range(CFG["n_multiline"]):
    rng = R(77000 + i * 3571)
    lines = [random_line(rng) for _ in range(rng.randint(2, 4))]
    gray = render_multiline(lines, rng, force_surface=SURFACES[i % len(SURFACES)])
    gt = "\n".join(lines)
    pred = predict_document(gray)
    cer, wa, sf = score(gt, pred)
    multi.append(dict(gt=gt, pred=pred, cer=cer, word_acc=wa, space_f1=sf,
                      lines_ok=float(pred.count("\n") + 1 == len(lines)),
                      detected=pred.count("\n") + 1))
multi_m = dict(
    cer=float(np.mean([m["cer"] for m in multi])),
    word_acc=float(np.mean([m["word_acc"] for m in multi])),
    space_f1=float(np.mean([m["space_f1"] for m in multi])),
    lines_ok=float(np.mean([m["lines_ok"] for m in multi])),
)
print("МНОГОСТРОЧНЫЙ ТЕКСТ   CER=%.4f  word_acc=%.3f  space_F1=%.3f  lines_ok=%.3f  (n=%d)"
      % (multi_m["cer"], multi_m["word_acc"], multi_m["space_f1"], multi_m["lines_ok"], len(multi)))


# %% [markdown]
# ## 6. Демо, отчёт, сохранение модели
# %%
rng = R(20261109)
demo_lines = ["The Medieval Warrior 2026", "level 7 boss fight", "TPU v5e 1 reads 42 words"]
demo = render_multiline(demo_lines, rng, force_surface="stone")
demo_pred = predict_document(demo)
demo.save(OUT / "demo_multiline.png")
print("--- ДЕМО: камень, 3 строки ---")
print("Эталон :", repr("\n".join(demo_lines)))
print("Модель :", repr(demo_pred))
print("Сегментировано строк:", len(segment_lines(demo)))
for s in SURFACES:
    r = R(4242 + len(s))
    txt = random_line(r)
    print("  %-11s %-34r -> %r" % (s, txt, read_line(render_line(txt, r, force_surface=s))))

report = dict(
    device=DEVICE_INFO, mode="smoke" if SMOKE else "full", config=CFG, charset=CHARS,
    params=sum(p.numel() for p in model.parameters()), test=test_m,
    per_surface=by_surface, multiline=multi_m,
    demo=dict(gt="\n".join(demo_lines), pred=demo_pred),
    examples=[dict(surface=r[0], gt=r[1], pred=r[2]) for r in test_rows[:12]],
    multiline_examples=multi[:5],
)
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
save_ckpt(OUT / "char_cnn.pt")
print("\nГОТОВО. report.json, char_cnn.pt, demo_multiline.png ->", OUT)
print("Итог: CER=%.4f | word_acc=%.3f | space_F1=%.3f | многострочность lines_ok=%.3f"
      % (test_m["cer"], test_m["word_acc"], test_m["space_f1"], multi_m["lines_ok"]))
