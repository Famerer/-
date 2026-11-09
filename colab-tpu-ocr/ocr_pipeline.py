# %% [markdown]
# # OCR на TPU v5e-1: буквы, цифры, слова, пробелы, многострочность, разные поверхности
#
# Полностью самодостаточный ноутбук. Ничего не скачивается: модель обучается на
# синтетических данных, которые генерируются прямо здесь.
#
# Что умеет:
# - читать **латиницу в обоих регистрах и цифры**;
# - сохранять **пробелы между словами** (отдельный класс + геометрическая проверка промежутков);
# - читать **несколько строк** (сегментация строк + распознавание каждой);
# - работать на **разных поверхностях**: бумага, градиент/блик, школьная доска, камень, дерево,
#   плюс шум, размытие, поворот, изменение яркости и контраста.
#
# Архитектура: CNN -> BiGRU -> CTC. Ускоритель выбирается автоматически:
# TPU (torch_xla) -> GPU (CUDA) -> CPU. На TPU v5e-1 всё считается без изменений кода.
#
# **Запуск:** Runtime -> Change runtime type -> TPU v5e-1, затем Runtime -> Run all.
# Для быстрой локальной проверки: `OCR_SMOKE=1 python ocr_pipeline.py`.
#
# %% [markdown]
# ## 1. Настройки и ускоритель
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

CFG = dict(
    img_h=48,
    img_w=256 if SMOKE else 320,
    n_train=1600 if SMOKE else 24000,
    n_val=240 if SMOKE else 1600,
    n_test=180 if SMOKE else 1200,
    n_multiline=24 if SMOKE else 160,
    epochs=8 if SMOKE else 14,
    batch=32 if SMOKE else 128,
    c=24 if SMOKE else 48,
    hidden=96 if SMOKE else 192,
    lr=2e-3,
    small_vocab=SMOKE,
)
OUT = Path(os.environ.get("OCR_OUT", str(Path(__file__).resolve().parent / "out")))
OUT.mkdir(parents=True, exist_ok=True)

XM = None


def detect_device():
    global XM
    try:
        import torch_xla.core.xla_model as xm
        XM = xm
        dev = xm.xla_device()
        return dev, "tpu", "TPU (torch_xla / PJRT): " + str(dev)
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

# CPU-рантайм в Colab: автоматически урезаем масштаб, чтобы обучение не шло часами.
if not SMOKE and DEVICE_KIND == "cpu":
    print("Внимание: выдан CPU-рантайм, автоматически уменьшаю объём обучения")
    CFG.update(img_w=256, n_train=2500, n_val=300, n_test=300,
               n_multiline=30, epochs=4, batch=32, c=24, hidden=96)
if os.environ.get("OCR_QUICK") == "1":
    CFG.update(n_train=640, n_val=60, n_test=60, n_multiline=8, epochs=3, batch=32)

print("Ускоритель:", DEVICE_INFO)
print("Режим:", "SMOKE (локальная проверка)" if SMOKE else "ПОЛНОЕ ОБУЧЕНИЕ")
print("Конфиг:", json.dumps({k: v for k, v in CFG.items() if k != "small_vocab"}))
print("Выходная папка:", OUT)

# %%
CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
IDX2CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}
NCLS = len(CHARS) + 1          # 0 = CTC blank
BLANK = 0
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
print("Найдено шрифтов:", len(FONTS))


class R:
    """Один ГПСЧ на生成: python-random для скаляров, numpy для массивов."""

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
    """Разные поверхности + освещение. Возвращает RGB и флаг 'тёмный фон'."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lf = lowfreq(rng, h, w, 12)
    n = rng.normal(0, 1, (h, w))
    dark = False
    if kind == "paper":
        base = 236 + lf * 0.10 + n * 3
    elif kind == "gradient":
        base = 228 - 75 * (xx / w) + lf * 0.12 + n * 4
    elif kind == "glare":
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        base = 232 + lf * 0.10 + n * 3
        base = base + 85 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.25) ** 2)))
    elif kind == "chalkboard":
        base = 44 + lf * 0.30 + n * 6
        dark = True
    elif kind == "stone":
        base = 152 + lf * 0.55 + n * 10
    elif kind == "wood":
        base = 174 + 18 * np.sin(yy * 0.22 + xx * 0.012) + lf * 0.35 + n * 7
    else:
        base = 230 + lf * 0.1 + n * 4
    base = np.clip(base, 0, 255)
    tint = rng.normal(0, 5, 3).astype(np.float32)
    rgb = np.clip(base[..., None] + tint[None, None, :], 0, 255).astype(np.uint8)
    return rgb, dark


def _font_for(rng, text, w):
    if not FONTS:
        try:
            return ImageFont.load_default(size=30), 30
        except TypeError:
            return ImageFont.load_default(), 12
    fp = rng.choice(FONTS)
    size = rng.randint(26, 36)
    font = ImageFont.truetype(fp, size)
    def est(f, s):
        return sum((s * 0.5 + 2) if ch == " " else (f.getlength(ch) + 1.4) for ch in text)
    while est(font, size) > w - 24 and size > 16:
        size -= 2
        font = ImageFont.truetype(fp, size)
    return font, size


def _draw_text(d, text, font, x, y, size, rng):
    for ch in text:
        if ch == " ":
            x += size * 0.5 + rng.randint(0, 4)
            continue
        sw = 1 if rng.random() < 0.35 else 0
        d.text((x, y), ch, font=font, fill=255, anchor="lt", stroke_width=sw, stroke_fill=255)
        x += font.getlength(ch) + rng.uniform(0.4, 2.0)
    return x


def _composite(mask, rgb, dark, rng, h, w):
    a = np.asarray(mask).astype(np.float32) / 255.0
    a *= np.clip(1 + lowfreq(rng, h, w, 7) / 420, 0.72, 1.28)
    a = np.clip(a, 0, 1)
    if dark:
        ink = np.clip(rng.normal(228, 7, 3), 0, 255).astype(np.float32)
    else:
        ink = np.clip(rng.normal(26, 9, 3), 0, 255).astype(np.float32)
    img = rgb.astype(np.float32) * (1 - a[..., None]) + ink[None, None, :] * a[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _augment(im, rng, rgb):
    fill = tuple(int(v) for v in rgb[0, 0])
    if rng.random() < 0.40:
        im = im.rotate(rng.uniform(-1.6, 1.6), resample=Image.BILINEAR, fillcolor=fill)
    if rng.random() < 0.45:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.85)))
    arr = np.asarray(im).astype(np.float32)
    arr = np.clip((arr - 127.5) * rng.uniform(0.82, 1.25) + 127.5 + rng.uniform(-14, 14), 0, 255)
    arr = arr + rng.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L")


def random_word(rng):
    if CFG["small_vocab"]:
        return rng.choice(SMOKE_WORDS)
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(2, 8)))


def random_line(rng):
    words = [random_word(rng) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.22:
        words = [w.upper() for w in words]
    return " ".join(words)


def render_line(text, rng, force_surface=None):
    w, h = CFG["img_w"], CFG["img_h"]
    kind = force_surface or rng.choice(SURFACES)
    rgb, dark = make_surface(rng, w, h, kind)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    font, size = _font_for(rng, text, w)
    tw = sum((size * 0.5 + 2) if ch == " " else (font.getlength(ch) + 1.4) for ch in text)
    x = max(6, min(w - tw - 6, rng.randint(8, max(9, w - 10))))
    y = (h - size) // 2 - 2 + rng.randint(-2, 2)
    _draw_text(d, text, font, x, y, size, rng)
    return _augment(_composite(mask, rgb, dark, rng, h, w), rng, rgb)


def render_multiline(lines, rng, force_surface=None):
    w = CFG["img_w"]
    lh, pad = 64, 20
    h = pad * 2 + len(lines) * lh
    kind = force_surface or rng.choice(SURFACES)
    rgb, dark = make_surface(rng, w, h, kind)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for i, text in enumerate(lines):
        font, size = _font_for(rng, text, w)
        x = rng.randint(10, 34)
        y = pad + i * lh + (lh - size) // 2 - 2
        _draw_text(d, text, font, x, y, size, rng)
    return _augment(_composite(mask, rgb, dark, rng, h, w), rng, rgb)


def to_tensor(gray):
    a = np.asarray(gray.convert("L"), dtype=np.float32) / 255.0
    z = np.clip((a - a.mean()) / (a.std() + 1e-5), -3, 3) / 3.0
    return torch.from_numpy(z[None, None, :, :].copy())


# %% [markdown]
# ## 2. Датасет и модель (CNN + BiGRU + CTC)
# %%
class LineDataset(Dataset):
    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = R(self.seed + i * 7919)
        text = random_line(rng)
        img = render_line(text, rng)
        target = torch.tensor([CHAR2IDX[ch] for ch in text], dtype=torch.long)
        return to_tensor(img), target, text


def collate(batch):
    xs = torch.cat([b[0] for b in batch], 0)
    ys = torch.cat([b[1] for b in batch], 0)
    lens = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return xs, ys, lens, [b[2] for b in batch]


class CRNN(nn.Module):
    def __init__(self, ncls, c=48, hidden=192):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), (3, 1)),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
        )
        # Transformer вместо RNN: только matmul/LayerNorm/softmax,
        # то есть 100% совместимо с XLA/TPU (у RNN на TPU бывают неподдерживаемые операции).
        dim = c * 4
        pe = torch.zeros(1, 512, dim)
        pos = torch.arange(512).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * -math.log(10000.0) / dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4,
                                           dim_feedforward=hidden * 2,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, ncls)

    def forward(self, x):
        f = self.cnn(x).mean(dim=2).permute(0, 2, 1)   # [B, T, C]
        f = f + self.pe[:, :f.size(1), :]
        z = self.enc(f)
        return self.head(z).log_softmax(2)


model = CRNN(NCLS, CFG["c"], CFG["hidden"]).to(DEVICE)


def save_ckpt(path):
    # На TPU state_dict содержит XLA-тензоры: явно переносим на CPU перед сохранением.
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)


print("Параметров модели:", sum(p.numel() for p in model.parameters()))
print("Длина последовательности T = img_w/4 =", CFG["img_w"] // 4)


# %% [markdown]
# ## 3. Обучение
# %%
loader = DataLoader(LineDataset(CFG["n_train"], SEED + 1), batch_size=CFG["batch"],
                    shuffle=True, collate_fn=collate, num_workers=0, drop_last=False)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
best_loss = float("inf")
t0 = time.time()

for ep in range(1, CFG["epochs"] + 1):
    model.train()
    total, nb = 0.0, 0
    for xs, ys, lens, _ in loader:
        xs, ys, lens = xs.to(DEVICE), ys.to(DEVICE), lens.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        lp = model(xs)                                    # [B, T, NCLS]
        T = lp.size(1)
        inp_len = torch.full((xs.size(0),), T, dtype=torch.long, device=DEVICE)
        loss = F.ctc_loss(lp.transpose(0, 1).contiguous(), ys, inp_len, lens,
                          blank=BLANK, zero_infinity=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if DEVICE_KIND == "tpu":
            XM.mark_step()
        total = total + loss.detach()
        nb += 1
    sched.step()
    val = float(total.cpu()) / max(1, nb) if torch.is_tensor(total) else total / max(1, nb)
    if val < best_loss:
        best_loss = val
        save_ckpt(OUT / "ocr_crnn.pt")
    print("эпоха %2d/%d  CTC loss = %.4f  lr = %.2e  t = %.1fs"
          % (ep, CFG["epochs"], val, opt.param_groups[0]["lr"], time.time() - t0))

print("Обучение заняло %.1f c. Лучший чекпоинт: %s" % (time.time() - t0, OUT / "ocr_crnn.pt"))
model.load_state_dict(torch.load(OUT / "ocr_crnn.pt", map_location="cpu"))
model.to(DEVICE)


# %% [markdown]
# ## 4. Инференс: пробелы, сегментация строк, метрики
# %%
def decode(logprobs):
    path = logprobs[0].argmax(-1).cpu().numpy().tolist()
    pairs, prev = [], -1
    for t, idx in enumerate(path):
        if idx != prev and idx != BLANK:
            pairs.append((t, IDX2CHAR[idx]))
        prev = idx
    return "".join(ch for _, ch in pairs), pairs


def empty_runs(gray, min_len=None):
    arr = np.asarray(gray.convert("L")).astype(np.float32)
    bg = np.asarray(gray.convert("L").filter(ImageFilter.GaussianBlur(9))).astype(np.float32)
    ink = np.abs(arr - bg) > 20
    col = ink.sum(axis=0)
    thr = max(1.0, 0.12 * float(col.max()))
    empty = col < thr
    W = len(empty)
    if min_len is None:
        min_len = max(4, int(W * 0.018))
    runs, start = [], None
    for i, v in enumerate(empty):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len and start > 4 and i < W - 4:
                runs.append((start, i))
            start = None
    return runs, W


def enhance_spaces(text, pairs, gray, strength=True):
    """Геометрическая подстраховка: длинные пустые колонки = пробел."""
    if not strength or not pairs:
        return text
    runs, W = empty_runs(gray)
    T = CFG["img_w"] // 4
    out = list(text)
    ins = set()
    for s, e in runs:
        gf = ((s + e) / 2) / W * T
        k = sum(1 for t, _ in pairs if t < gf)
        if k <= 0 or k >= len(out):
            continue
        if out[k - 1] == " " or out[k] == " ":
            continue
        ins.add(k)
    for k in sorted(ins, reverse=True):
        out.insert(k, " ")
    return "".join(out)


@torch.no_grad()
def predict_line(gray, use_gaps=True):
    model.eval()
    lp = model(to_tensor(gray).to(DEVICE))
    text, pairs = decode(lp)
    return enhance_spaces(text, pairs, gray, use_gaps)


def segment_lines(gray):
    arr = np.asarray(gray.convert("L")).astype(np.float32)
    bg = np.asarray(gray.convert("L").filter(ImageFilter.GaussianBlur(11))).astype(np.float32)
    ink = np.abs(arr - bg) > 22
    active = ink.sum(axis=1) > max(3, 0.02 * arr.shape[1])
    groups, start = [], None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            groups.append([start, i - 1])
            start = None
    if start is not None:
        groups.append([start, len(active) - 1])
    merged = []
    for a, b in groups:
        if merged and a - merged[-1][1] < 6:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < 9:
            continue
        a, b = max(0, a - 5), min(arr.shape[0] - 1, b + 5)
        out.append(gray.crop((0, a, arr.shape[1], b + 1)))
    return out


def prep_crop(crop):
    bw, bh = crop.size
    nw = min(CFG["img_w"], max(1, int(bw * CFG["img_h"] / bh)))
    r = crop.resize((nw, CFG["img_h"]), Image.BILINEAR)
    bg = int(np.median(np.asarray(crop.convert("L"))))
    canvas = Image.new("L", (CFG["img_w"], CFG["img_h"]), bg)
    canvas.paste(r, (0, 0))
    return canvas


@torch.no_grad()
def predict_document(gray, use_gaps=True):
    bands = segment_lines(gray)
    return "\n".join(predict_line(prep_crop(b), use_gaps) for b in bands) if bands else ""


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
    exact_words = g.split() == p.split()
    gs = {i for i, ch in enumerate(g) if ch == " "}
    ps = {i for i, ch in enumerate(p) if ch == " "}
    tp = len(gs & ps)
    sf1 = 2 * tp / max(1, len(gs) + len(ps))
    return cer, exact_words, sf1


def evaluate(use_gaps, n=None, seed=99001, per_surface=False, surface=None):
    n = n or CFG["n_val"]
    rows = []
    for i in range(n):
        rng = R(seed + i * 104729)
        surf = surface or (SURFACES[i % len(SURFACES)] if per_surface else None)
        text = random_line(rng)
        gray = render_line(text, rng, force_surface=surf)
        pred = predict_line(gray, use_gaps)
        rows.append((surf or "mixed", text, pred, *score(text, pred)))
    return rows


def summarize(rows, label):
    cer = float(np.mean([r[3] for r in rows]))
    wacc = float(np.mean([r[4] for r in rows]))
    sf1 = float(np.mean([r[5] for r in rows]))
    print("%-22s CER=%.4f  word_acc=%.3f  space_F1=%.3f  (n=%d)"
          % (label, cer, wacc, sf1, len(rows)))
    return dict(cer=cer, word_acc=wacc, space_f1=sf1, n=len(rows))


# %% [markdown]
# ## 5. Автовыбор режима пробелов и полная проверка
# %%
rows_off = evaluate(False, n=min(CFG["n_val"], 240))
rows_on = evaluate(True, n=min(CFG["n_val"], 240))
m_off, m_on = summarize(rows_off, "CTC без геометрии"), summarize(rows_on, "CTC + геометрия")
USE_GAPS = m_on["cer"] < m_off["cer"] - 0.005
print("Автовыбор режима пробелов:", "CTC + геометрия" if USE_GAPS else "только CTC")

test_rows = evaluate(USE_GAPS, n=CFG["n_test"], seed=555001)
test_m = summarize(test_rows, "ТЕСТ (смесь)")
by_surface = {}
for idx, s in enumerate(SURFACES):
    sub = evaluate(USE_GAPS, n=60, seed=6000 + idx * 101, surface=s)
    by_surface[s] = summarize(sub, "  поверхность: " + s)

multi = []
for i in range(CFG["n_multiline"]):
    rng = R(77000 + i * 3571)
    lines = [random_line(rng) for _ in range(rng.randint(2, 4))]
    gray = render_multiline(lines, rng, force_surface=SURFACES[i % len(SURFACES)])
    gt = "\n".join(lines)
    pred = predict_document(gray, USE_GAPS)
    cer, wa, sf = score(gt, pred)
    line_ok = pred.count("\n") + 1 == len(lines)
    multi.append(dict(gt=gt, pred=pred, cer=cer, word_acc=wa, space_f1=sf, lines_ok=line_ok))
multi_m = dict(
    cer=float(np.mean([m["cer"] for m in multi])),
    word_acc=float(np.mean([m["word_acc"] for m in multi])),
    space_f1=float(np.mean([m["space_f1"] for m in multi])),
    lines_ok=float(np.mean([m["lines_ok"] for m in multi])),
)
print("МНОГОСТРОЧНЫЙ ТЕКСТ   CER=%.4f  word_acc=%.3f  space_F1=%.3f  lines_ok=%.3f  (n=%d)"
      % (multi_m["cer"], multi_m["word_acc"], multi_m["space_f1"], multi_m["lines_ok"], len(multi)))


# %% [markdown]
# ## 6. Демо и сохранение результатов
# %%
rng = R(20261109)
demo_lines = ["The Medieval Warrior 2026", "level 7 boss fight", "TPU v5e 1 reads 42 words"]
demo = render_multiline(demo_lines, rng, force_surface="stone")
demo_pred = predict_document(demo, USE_GAPS)
demo.save(OUT / "demo_multiline.png")
print("--- ДЕМО: каменная поверхность, 3 строки ---")
print("Эталон :", repr("\n".join(demo_lines)))
print("Модель :", repr(demo_pred))
print("Сегментировано строк:", len(segment_lines(demo)))

for s in SURFACES:
    rng = R(4242 + len(s))
    txt = random_line(rng)
    pred = predict_line(render_line(txt, rng, force_surface=s), USE_GAPS)
    print("  %-11s %-34r -> %r" % (s, txt, pred))

report = dict(
    device=DEVICE_INFO, mode="smoke" if SMOKE else "full", config=CFG,
    charsets=CHARS, params=sum(p.numel() for p in model.parameters()),
    test=test_m, per_surface=by_surface, multiline=multi_m,
    space_mode="ctc+geometry" if USE_GAPS else "ctc",
    demo=dict(gt="\n".join(demo_lines), pred=demo_pred),
    examples=[dict(surface=r[0], gt=r[1], pred=r[2]) for r in test_rows[:12]],
)
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
save_ckpt(OUT / "ocr_crnn.pt")
print("\nГОТОВО. report.json, ocr_crnn.pt, demo_multiline.png ->", OUT)
print("Итог: CER=%.4f | word_acc=%.3f | space_F1=%.3f | многострочность lines_ok=%.3f"
      % (test_m["cer"], test_m["word_acc"], test_m["space_f1"], multi_m["lines_ok"]))
