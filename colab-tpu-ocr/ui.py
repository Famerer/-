"""Минимальная UI-автоматизация Windows через ctypes + PIL (без Add-Type)."""
import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

VK = {
    "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "enter": 0x0D, "esc": 0x1B,
    "tab": 0x09, "space": 0x20, "back": 0x08, "up": 0x26, "down": 0x28,
    "left": 0x25, "right": 0x27, "home": 0x24, "end": 0x23,
    **{"f%d" % i: 0x6F + i for i in range(1, 13)},
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(i): 0x30 + i for i in range(10)},
}


def windows():
    out = []

    def cb(h, _):
        n = user32.GetWindowTextLengthW(h)
        if n and user32.IsWindowVisible(h):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(h, buf, n + 1)
            out.append((h, buf.value))
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return out


def activate(substr):
    for h, title in windows():
        if substr.lower() in title.lower():
            user32.ShowWindow(h, 9)          # SW_RESTORE
            user32.SetForegroundWindow(h)
            time.sleep(0.4)
            print("activated:", title)
            return True
    print("NOT FOUND:", substr)
    return False


def click(x, y):
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.12)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    print("click", x, y)


def key(combo):
    parts = combo.lower().split("+")
    codes = [VK.get(p, VK.get(p.upper())) for p in parts]
    if any(c is None for c in codes):
        raise KeyError(combo)
    for c in codes:
        user32.keybd_event(c, 0, 0, 0)
    for c in reversed(codes):
        user32.keybd_event(c, 0, 0x0002, 0)
    print("key", combo)


def shot(path):
    from PIL import ImageGrab
    im = ImageGrab.grab()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print("shot", path, im.size)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "windows":
        for h, t in windows():
            print(h, t)
    elif cmd == "activate":
        activate(sys.argv[2])
    elif cmd == "click":
        click(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "key":
        key(sys.argv[2])
    elif cmd == "shot":
        shot(sys.argv[2])
    elif cmd == "seq":
        # seq <activate-substr> <delay> <action> <action>...  action = k:ctrl+f9 | c:x,y | s:path
        activate(sys.argv[2])
        time.sleep(float(sys.argv[3]))
        for a in sys.argv[4:]:
            kind, val = a.split(":", 1)
            if kind == "k":
                key(val)
            elif kind == "c":
                x, y = val.split(",")
                click(int(x), int(y))
            elif kind == "s":
                shot(val)
            elif kind == "w":
                time.sleep(float(val))
                print("wait", val)
            time.sleep(0.6)
