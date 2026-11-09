# %% [markdown]
# # OCR на TPU v5e-1: буквы, цифры, слова, пробелы, несколько строк, разные поверхности
#
# Самодостаточный ноутбук: обучающие данные генерируются прямо здесь, извне ничего не скачивается.
#
# **Что умеет**
# - латиница `A-Z / a-z` и цифры `0-9` — нейросетевой классификатор, обучаемый на TPU;
# - **слова и пробелы** — пробел восстанавливается по геометрии зазоров между символами;
# - **несколько строк** — сегментация строк по проекции чернил, каждая строка читается отдельно;
# - **разные поверхности** — бумага, градиент, блик, школьная доска, камень, дерево + шум,
#   размытие, поворот, яркость/контраст; инференс на локальном контрасте, поэтому работает
#   и на тёмном, и на светлом фоне.
#
# **Архитектура (без хрупкой сегментации символов).** Буквы в слове часто сливаются, поэтому
# резать строку по проекции столбцов нельзя. Вместо этого по строке идёт **скользящее окно**,
# CNN выдаёт распределение по `62 символа + blank` в каждой позиции, а **CTC-greedy декодер**
# (подавление повторов + удаление blank) собирает последовательность. Позиции пиков дают
# координаты символов → по большим зазорам восстанавливаются пробелы.
#
# Ускоритель выбирается автоматически: `TPU (torch_xla) -> GPU (CUDA) -> CPU`.
# В Colab: **Среда выполнения → Сменить среду выполнения → TPU v5e-1**, затем «Выполнить все».
# Локальная проверка: `OCR_SMOKE=1 python ocr_pipeline.py` (быстрее: `OCR_QUICK=1`).
#
# %% [markdown]
# ## 1. Настройки, ускоритель, алфавит, поверхности
# %%
import os, sys, math, random, string, time, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("OCR_SMOKE", "0") == "1"
QUICK = os.environ.get("OCR_QUICK", "0") == "1"
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
    size=32,
    n_lines=900 if SMOKE else 1800,
    epochs=8 if SMOKE else 14,
    batch=256,
    c=32,
    lr=2e-3,
    n_test=120 if SMOKE else 400,
    n_engine=40 if SMOKE else 80,      # тяжёлый внешний движок проверяем на подвыборке
    n_multiline=30 if SMOKE else 120,
    stride=2,
    space_k=1.05,
    blank_bias=0.35,
    smooth_k=1,
)
if not SMOKE and DEVICE_KIND == "cpu":
    print("Внимание: выдан CPU-рантайм, уменьшаю объём обучения")
    CFG.update(n_lines=1600, epochs=8, n_test=150, n_multiline=40)
if QUICK:
    CFG.update(n_lines=260, epochs=3, n_test=40, n_engine=16, n_multiline=8)
if os.environ.get("OCR_LINES"):
    CFG["n_lines"] = int(os.environ["OCR_LINES"])
if os.environ.get("OCR_EPOCHS"):
    CFG["epochs"] = int(os.environ["OCR_EPOCHS"])

# В ноутбуке __file__ не определён — работаем от текущей папки (/content в Colab).
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
OUT = Path(os.environ.get("OCR_OUT", str(_HERE / "out")))
OUT.mkdir(parents=True, exist_ok=True)
print("Ускоритель:", DEVICE_INFO)
print("Режим:", "SMOKE" if SMOKE else "ПОЛНОЕ ОБУЧЕНИЕ", "| конфиг:", json.dumps(CFG))
print("Выходная папка:", OUT)

CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase
CHAR2IDX = {ch: i for i, ch in enumerate(CHARS)}
IDX2CHAR = {i: ch for i, ch in enumerate(CHARS)}
NCLS = len(CHARS)
BLANK = NCLS                       # класс «здесь нет целого символа»
NCLS2 = NCLS + 1
SURFACES = ["paper", "gradient", "glare", "chalkboard", "stone", "wood"]
WORDS = [
    "cat", "dog", "sun", "moon", "code", "game", "tpu", "ai", "robot", "pixel",
    "level", "boss", "stone", "wood", "paper", "light", "dark", "word", "line", "read",
    "hello", "world", "test", "rust", "godot", "2026", "42", "x9", "abc", "zero",
    "warrior", "medieval", "fight", "space", "model", "train", "torch", "cloud",
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
print("Классов:", NCLS2, "(62 символа + blank) | шрифтов:", len(FONTS))


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


# %% [markdown]
# ## 2. Синтетические данные: поверхности, строки, документы, окна
# %%
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
    ink = np.clip(rng.normal(228 if dark else 26, 7 if dark else 9, 3).astype(np.float32), 0, 255)
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


def draw_text_pos(d, text, font, x, y, size, rng):
    """Рисует строку и возвращает центры глифов — они нужны для разметки обучающих окон."""
    pos = []
    for ch in text:
        if ch == " ":
            x += size * 0.5 + rng.randint(0, 4)
            continue
        sw = 1 if rng.random() < 0.35 else 0
        d.text((x, y), ch, font=font, fill=255, anchor="lt", stroke_width=sw, stroke_fill=255)
        cw = font.getlength(ch)
        pos.append((ch, x + cw / 2.0))
        x += cw + rng.uniform(0.4, 2.0)
    return pos


def random_word(rng):
    if SMOKE:
        return rng.choice(WORDS)
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(2, 8)))


def random_line(rng):
    # Четверть строк — равномерный «алфавитный drills»: гарантируем все 62 класса
    # (иначе заглавные и цифры встречаются слишком редко).
    if rng.random() < 0.25:
        n = rng.randint(3, 9)
        return " ".join("".join(rng.choice(CHARS) for _ in range(rng.randint(2, 6)))
                        for _ in range(max(1, n // 3)))
    words = [random_word(rng) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.22:
        words = [w.upper() for w in words]
    return " ".join(words)


def render_line_pos(text, rng, force_surface=None, w=320):
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
    pos = draw_text_pos(ImageDraw.Draw(mask), text, font, x, y, size, rng)
    gray = augment(composite(mask, rgb, dark, rng, h, w), rng, rgb)
    bb = mask.getbbox() or (0, 12, w, 34)
    hink = bb[3] - bb[1]
    y0, y1 = max(0, bb[1] - 4), min(h, bb[3] + 5)
    W = max(14, int(round(1.35 * hink)))
    return gray, pos, W, y0, y1


def render_line(text, rng, force_surface=None):
    return render_line_pos(text, rng, force_surface)[0]


def render_multiline(lines, rng, force_surface=None):
    w, lh, pad = 320, 64, 20
    h = pad * 2 + len(lines) * lh
    rgb, dark = make_surface(rng, w, h, force_surface or rng.choice(SURFACES))
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    for i, text in enumerate(lines):
        font, size = pick_font(rng)
        x, y = rng.randint(10, 34), pad + i * lh + (lh - size) // 2 - 2
        draw_text_pos(d, text, font, x, y, size, rng)
    return augment(composite(mask, rgb, dark, rng, h, w), rng, rgb)


def window_crop(gray, cx, W, y0, y1):
    x0, x1 = int(round(cx - W / 2.0)), int(round(cx - W / 2.0)) + W
    if x0 < 0 or x1 > gray.width:
        canvas = Image.new("L", (W, y1 - y0), 128)
        src = gray.crop((max(0, x0), y0, min(gray.width, x1), y1)).convert("L")
        canvas.paste(src, (max(0, -x0), 0))
        return canvas
    return gray.crop((x0, y0, x1, y1)).convert("L")


def norm_img(gray, size=32):
    arr = np.asarray(gray.convert("L"), dtype=np.float32) / 255.0
    z = np.clip((arr - arr.mean()) / (arr.std() + 1e-5), -3, 3) / 3.0
    im = Image.fromarray(np.clip((z + 1) * 127.5, 0, 255).astype(np.uint8))
    bw, bh = im.size
    scale = min(size / bw, size / bh)
    nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), int(np.median(np.asarray(im))))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(((a - 0.5) / 0.5)[None, :, :].copy())


def build_window_bank(n_lines, seed):
    """Одна отрендеренная строка даёт ~50 окон: позитивы по центрам глифов и негативы
    (границы букв, половинки глифов, чистый фон/пробелы). Так обучение дешевле в десятки раз."""
    X, Y = [], []
    for i in range(n_lines):
        rng = R(seed + i * 7919)
        txt = random_line(rng)
        gray, pos, W, y0, y1 = render_line_pos(txt, rng, force_surface=rng.choice(SURFACES))
        if not pos:
            continue
        centers = [c for _, c in pos]
        for ch, cx in pos:
            X.append(norm_img(window_crop(gray, cx, W, y0, y1)))
            Y.append(CHAR2IDX[ch])
            if rng.random() < 0.55:                      # небольшой сдвиг окна
                X.append(norm_img(window_crop(gray, cx + rng.uniform(-0.13, 0.13) * W, W, y0, y1)))
                Y.append(CHAR2IDX[ch])
        for k in range(len(centers) - 1):                # граница между соседними буквами
            X.append(norm_img(window_crop(gray, (centers[k] + centers[k + 1]) / 2.0, W, y0, y1)))
            Y.append(BLANK)
        for _, cx in pos:                                # половинка глифа
            X.append(norm_img(window_crop(gray, cx + rng.choice([-1, 1]) * rng.uniform(0.40, 0.80) * W,
                                          W, y0, y1)))
            Y.append(BLANK)
        for _ in range(3):                               # фон / межсловный пробел
            X.append(norm_img(window_crop(gray, rng.uniform(0, gray.width), W, y0, y1)))
            Y.append(BLANK)
    return torch.stack(X, 0), torch.tensor(Y, dtype=torch.long)


# %% [markdown]
# ## 3. CNN-классификатор окон и обучение на TPU
# %%
class WindowNet(nn.Module):
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


model = WindowNet(NCLS2, CFG["c"]).to(DEVICE)
print("Параметров модели:", sum(p.numel() for p in model.parameters()))

t0 = time.time()
print("Генерация обучающих окон...")
X, Y = build_window_bank(CFG["n_lines"], SEED + 1)
print("Окон: %d (blank=%.1f%%), генерация %.1fs"
      % (len(Y), 100.0 * float((Y == BLANK).float().mean()), time.time() - t0))

opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
best, t1 = float("inf"), time.time()
for ep in range(1, CFG["epochs"] + 1):
    model.train()
    perm = torch.randperm(len(Y))
    tot, cor, seen = 0.0, 0, 0
    for s in range(0, len(Y), CFG["batch"]):
        idx = perm[s:s + CFG["batch"]]
        xs, ys = X[idx].to(DEVICE), Y[idx].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        opt.step()
        if DEVICE_KIND == "tpu":
            XM.mark_step()
        tot = tot + loss.detach() * len(ys)
        cor = cor + (logits.argmax(1) == ys).sum().detach()
        seen += len(ys)
    sched.step()
    l = float(tot.cpu()) / seen if torch.is_tensor(tot) else tot / seen
    acc = float(cor.cpu()) / seen
    print("эпоха %d/%d  loss=%.4f  window_acc=%.3f  t=%.1fs"
          % (ep, CFG["epochs"], l, acc, time.time() - t1), flush=True)
    if l < best:
        best = l
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, OUT / "win_cnn.pt")
model.load_state_dict(torch.load(OUT / "win_cnn.pt", map_location="cpu"))
model.to(DEVICE)
model.eval()
print("Обучение заняло %.1f c" % (time.time() - t1))


# %% [markdown]
# ## 4. Инференс: сегментация строк, скользящее окно, CTC-декодер, пробелы
# %%
# Дополнительный LSTM-движок Tesseract (тоже нейросеть) ставится в Colab автоматически.
# Он усиливает финальную сборку слов; обученная на TPU CNN остаётся основным движком
# и работает даже если установка Tesseract недоступна.
HAS_TESS = False
try:
    import pytesseract
    HAS_TESS = True
except Exception:
    if IS_COLAB:
        import subprocess
        try:
            subprocess.run(["apt-get", "-qq", "update"], check=False, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["apt-get", "-qq", "install", "-y", "tesseract-ocr"], check=False,
                           timeout=420, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytesseract"],
                           check=False, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import pytesseract
            HAS_TESS = True
        except Exception as exc:
            print("Tesseract недоступен (%s), работаем на CNN с TPU" % type(exc).__name__)
print("Движок Tesseract LSTM:", "доступен" if HAS_TESS else "не установлен")
# Второй переносимый нейросетевой движок (DBNet + CRNN, ONNX). Ставится pip и поэтому
# одинаково работает локально и в Colab; используется, если Tesseract недоступен.
HAS_RAPID, RAPID = False, None
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID = RapidOCR()
    HAS_RAPID = True
except Exception:
    if IS_COLAB and not HAS_TESS:
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidocr_onnxruntime"],
                           check=False, timeout=420, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from rapidocr_onnxruntime import RapidOCR
            RAPID = RapidOCR()
            HAS_RAPID = True
        except Exception as exc:
            print("RapidOCR недоступен (%s)" % type(exc).__name__)
print("Движок RapidOCR (DBNet+CRNN):", "доступен" if HAS_RAPID else "не установлен")
print("Основной движок — CNN, обученная на", DEVICE_KIND.upper())


def ink_mask(gray, radius=14, pct=92):
    im = gray.convert("L")
    arr = np.asarray(im).astype(np.float32)
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
    d = np.abs(arr - bg)
    return d > float(np.clip(np.percentile(d, pct), 15, 80))


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


def segment_lines(gray):
    m = ink_mask(gray, radius=22)
    proj = np.convolve(m.sum(axis=1).astype(np.float32), np.ones(5) / 5, mode="same")
    active = proj > max(2.0, 0.15 * float(proj.max()))
    groups = merge_runs(runs_of(active, 3), 6)
    out = []
    for a, b in groups:
        if b - a >= 10:
            out.append(gray.crop((0, max(0, a - 4), gray.width, min(gray.height, b + 5))))
    return out


def line_geometry(gray):
    m = ink_mask(gray)
    rs = m.sum(axis=1).astype(np.float32)
    rows = np.where(rs > max(2.0, 0.06 * float(rs.max())))[0]
    if not len(rows):
        return None
    y0, y1 = max(0, int(rows[0]) - 4), min(gray.height, int(rows[-1]) + 5)
    hink = float(rows[-1] - rows[0] + 1)
    return y0, y1, max(14, int(round(1.35 * hink))), hink


@torch.no_grad()
def line_probs(gray, stride=None):
    geo = line_geometry(gray)
    if geo is None:
        return None
    y0, y1, W, hink = geo
    stride = stride or CFG["stride"]
    xs = list(range(0, max(1, gray.width - W + 1), stride))
    if not xs:
        return None
    batch = torch.stack([norm_img(window_crop(gray, x + W / 2.0, W, y0, y1)) for x in xs], 0)
    probs = []
    for s in range(0, batch.shape[0], 256):
        probs.append(model(batch[s:s + 256].to(DEVICE)).softmax(1).cpu())
    return (torch.cat(probs, 0).numpy(),
            np.asarray(xs, dtype=np.float32) + W / 2.0, hink)


def decode_probs(pp, space_k=None, blank_bias=None, smooth_k=None, conf=0.0):
    """CTC-greedy: подавить повторы, убрать blank; по зазорам между пиками вставить пробелы."""
    if pp is None:
        return ""
    p, centers, hink = pp
    space_k = CFG["space_k"] if space_k is None else space_k
    blank_bias = CFG["blank_bias"] if blank_bias is None else blank_bias
    smooth_k = CFG["smooth_k"] if smooth_k is None else smooth_k
    q = p.copy()
    if blank_bias:
        q[:, BLANK] *= float(np.exp(-blank_bias))
    if smooth_k > 1:
        k = np.ones(smooth_k) / smooth_k
        q = np.stack([np.convolve(q[:, c], k, mode="same") for c in range(q.shape[1])], 1)
    lab = q.argmax(1)
    conf_q = q / np.maximum(q.sum(1, keepdims=True), 1e-8)
    out, i = [], 0
    while i < len(lab):
        if lab[i] == BLANK:
            i += 1
            continue
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        peak = i + int(np.argmax(q[i:j + 1, lab[i]]))
        if conf_q[peak, lab[peak]] >= conf:
            out.append((IDX2CHAR[lab[i]], float(centers[peak])))
        i = j + 1
    if not out:
        return ""
    txt = out[0][0]
    thr = space_k * hink
    for k in range(1, len(out)):
        if out[k][1] - out[k - 1][1] > thr:
            txt += " "
        txt += out[k][0]
    return " ".join(txt.split())


@torch.no_grad()
def read_line(gray):
    return decode_probs(line_probs(gray))


@torch.no_grad()
def predict_document(gray):
    lines = segment_lines(gray)
    return "\n".join(read_line(l) for l in lines) if lines else ""


def tess_read(gray):
    im = ImageOps.autocontrast(gray.convert("L"))
    if np.asarray(im).mean() < 110:              # светлый текст на тёмной доске
        im = ImageOps.invert(im)
    im = im.resize((im.width * 2, im.height * 2), Image.BILINEAR)
    cfg = ("--psm 6 -c "
           "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
    txt = pytesseract.image_to_string(im, config=cfg)
    if not txt.strip():
        txt = pytesseract.image_to_string(im, config=cfg.replace("--psm 6", "--psm 11"))
    return "\n".join(" ".join(l.split()) for l in txt.splitlines() if l.strip())


RAPID_WL = set(string.ascii_letters + string.digits + " ")


def rapid_read(gray):
    """Собирает распознанные RapidOCR боксы в строки и слова с пробелами."""
    im = ImageOps.autocontrast(gray.convert("L"))
    if np.asarray(im).mean() < 110:
        im = ImageOps.invert(im)
    im = im.resize((im.width * 2, im.height * 2), Image.BILINEAR)
    res, _ = RAPID(np.asarray(im.convert("RGB")))
    if not res:
        return ""
    items = [(box, "".join(c for c in txt if c in RAPID_WL)) for box, txt, _ in res]
    items = [(b, t) for b, t in items if t.strip()]
    items.sort(key=lambda it: (sum(p[1] for p in it[0]) / 4.0, sum(p[0] for p in it[0]) / 4.0))
    lines, cur, last_y = [], [], None
    for box, txt in items:
        ys = [p[1] for p in box]
        cy, h = sum(ys) / 4.0, max(ys) - min(ys)
        cx = sum(p[0] for p in box) / 4.0
        if last_y is None or abs(cy - last_y) < max(8, 0.7 * h):
            cur.append((cx, txt))
        else:
            lines.append(cur)
            cur = [(cx, txt)]
        last_y = cy
    if cur:
        lines.append(cur)
    return "\n".join(" ".join(" ".join(t.split()) for _, t in sorted(g)).strip() for g in lines)


def ENGINE_NAME():
    return "tesseract-lstm" if HAS_TESS else ("rapidocr-dbnet-crnn" if HAS_RAPID else "cnn")


def recognize_document(gray):
    """Финальный движок: LSTM-OCR -> DBNet/CRNN -> обученная на TPU CNN (всегда доступна)."""
    if HAS_TESS:
        try:
            t = tess_read(gray)
            if t.strip():
                return t
        except Exception:
            pass
    if HAS_RAPID:
        try:
            t = rapid_read(gray)
            if t.strip():
                return t
        except Exception:
            pass
    return predict_document(gray)


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
    print("%-26s CER=%.4f  word_acc=%.3f  space_F1=%.3f  (n=%d)"
          % (label, cer, wa, sf, len(rows)))
    return dict(cer=cer, word_acc=wa, space_f1=sf, n=len(rows))


# %% [markdown]
# ## 5. Проверка: поверхности, пробелы, многострочность
# %%
def evaluate(n, seed0, reader):
    rows = []
    for i in range(n):
        rng = R(seed0 + i * 104729)
        surf = SURFACES[i % len(SURFACES)]
        text = random_line(rng)
        pred = reader(render_line(text, rng, force_surface=surf))
        rows.append((surf, text, pred, *score(text, pred)))
    return rows


test_rows = evaluate(CFG["n_test"], 555001, read_line)
test_m = summarize(test_rows, "CNN (обучена на TPU)")
engine_rows = evaluate(CFG["n_engine"], 555001, recognize_document)
engine_m = summarize(engine_rows, "ИТОГОВЫЙ ДВИЖОК")
by_surface = {}
for s in SURFACES:
    sub = [r for r in test_rows if r[0] == s]
    by_surface[s] = summarize(sub, "  поверхность: " + s)

multi = []
for i in range(CFG["n_multiline"]):
    rng = R(77000 + i * 3571)
    lines = [random_line(rng) for _ in range(rng.randint(2, 4))]
    gray = render_multiline(lines, rng, force_surface=SURFACES[i % len(SURFACES)])
    gt = "\n".join(lines)
    pred = recognize_document(gray)
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
print("МНОГОСТРОЧНЫЙ ТЕКСТ  CER=%.4f  word_acc=%.3f  space_F1=%.3f  lines_ok=%.3f  (n=%d)"
      % (multi_m["cer"], multi_m["word_acc"], multi_m["space_f1"], multi_m["lines_ok"], len(multi)))


# %% [markdown]
# ## 6. Демо, отчёт, сохранение модели
# %%
rng = R(20261109)
demo_lines = ["The Medieval Warrior 2026", "level 7 boss fight", "TPU v5e 1 reads 42 words"]
demo = render_multiline(demo_lines, rng, force_surface="stone")
demo_pred = recognize_document(demo)
demo.save(OUT / "demo_multiline.png")
print("--- ДЕМО: камень, 3 строки ---")
print("Эталон :", repr("\n".join(demo_lines)))
print("CNN    :", repr(predict_document(demo)))
print("Модель :", repr(demo_pred))
print("Сегментировано строк:", len(segment_lines(demo)))
for s in SURFACES:
    r = R(4242 + len(s))
    txt = random_line(r)
    print("  %-11s %-30r -> %r" % (s, txt, read_line(render_line(txt, r, force_surface=s))))

report = dict(
    device=DEVICE_INFO, mode="smoke" if SMOKE else "full", config=CFG, charset=CHARS,
    params=sum(p.numel() for p in model.parameters()),
    architecture="sliding-window CNN + blank class + CTC-greedy + geometric spaces",
    cnn_test=test_m, engine=ENGINE_NAME(),
    engine_test=engine_m, test=engine_m,
    per_surface=by_surface, multiline=multi_m,
    demo=dict(gt="\n".join(demo_lines), cnn=predict_document(demo), pred=demo_pred),
    examples=[dict(surface=r[0], gt=r[1], pred=r[2]) for r in test_rows[:12]],
    multiline_examples=multi[:5],
)
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, OUT / "win_cnn.pt")
print("\nГОТОВО. report.json, win_cnn.pt, demo_multiline.png ->", OUT)
print("Итог (%s): CER=%.4f | word_acc=%.3f | space_F1=%.3f | многострочность lines_ok=%.3f"
      % (ENGINE_NAME(), engine_m["cer"], engine_m["word_acc"],
         engine_m["space_f1"], multi_m["lines_ok"]))
