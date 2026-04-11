import cv2
import numpy as np
import mediapipe as mp
import math
import time
from datetime import datetime

# ── MediaPipe Setup ──
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ── State ──
canvas       = None   # initialized after camera resolution is known
undo_stack   = []
brush_type   = "pen"
brush_color  = (255, 142, 108)   # BGR  (looks blue-ish on screen)
brush_size   = 8
brush_opacity = 0.9
last_x, last_y = 0, 0
is_drawing   = False
gesture_debounce = {}
color_index  = 0
brush_idx    = 0

BRUSH_TYPES = ["pen", "marker", "spray", "glow", "neon", "eraser"]

# BGR color cycle — 9 colors, 5 per row = 2 rows
COLOR_CYCLE = [
    (255, 142, 108),   # cornflower blue
    (255, 108, 180),   # pink
    (108, 255, 142),   # green
    (108, 233, 255),   # yellow
    (108, 140, 255),   # orange
    (255, 108, 196),   # purple
    (238, 255, 108),   # cyan
    (255, 255, 255),   # white
    (68,  68,  255),   # red
]

PANEL_W = 240   # sidebar width in pixels

# ── Toast ──
toast_msg   = ""
toast_until = 0.0

def show_toast(msg, duration=2.0):
    global toast_msg, toast_until
    toast_msg   = msg
    toast_until = time.time() + duration

# ── Canvas helpers ──
def save_snapshot():
    if len(undo_stack) > 20:
        undo_stack.pop(0)
    undo_stack.append(canvas.copy())

def undo():
    if undo_stack:
        canvas[:] = undo_stack.pop()
        show_toast("Undo!")
    else:
        show_toast("Nothing to undo")

def clear_canvas():
    save_snapshot()
    canvas[:] = 0
    show_toast("Canvas cleared")

def save_image():
    fname = f"airwrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(fname, canvas)
    show_toast(f"Saved: {fname}")
    print(f"[SAVED] {fname}")

# ── Geometry ──
def is_finger_up(lm, tip_id, pip_id):
    return lm.landmark[tip_id].y < lm.landmark[pip_id].y

def lm_dist(lm, a, b):
    ax, ay = lm.landmark[a].x, lm.landmark[a].y
    bx, by = lm.landmark[b].x, lm.landmark[b].y
    return math.hypot(ax - bx, ay - by)

# ── Gesture Detection ──
def detect_gesture(lm):
    index  = is_finger_up(lm, 8,  6)
    middle = is_finger_up(lm, 12, 10)
    ring   = is_finger_up(lm, 16, 14)
    pinky  = is_finger_up(lm, 20, 18)
    pinch  = lm_dist(lm, 4, 8) < 0.06
    count  = sum([index, middle, ring, pinky])

    thumb_up = (not index and not middle and not ring and not pinky
                and lm.landmark[4].y < lm.landmark[3].y)
    # 🤘 Rock On: index + pinky up, middle + ring down
    rock_on  = (index and not middle and not ring and pinky)
    # 🤙 Shaka: pinky up, index+middle+ring down, thumb extended sideways
    thumb_x   = lm.landmark[4].x
    wrist_x   = lm.landmark[0].x
    thumb_out = abs(thumb_x - wrist_x) > 0.1
    shaka     = (not index and not middle and not ring and pinky and thumb_out)

    if shaka:                                              return "SHAKA"
    if thumb_up:                                           return "THUMB_UP"
    if rock_on:                                            return "ROCK_ON"
    if count == 4:                                         return "PALM"
    if index and not middle and not ring and not pinky:    return "DRAW"
    if index and middle and not ring and not pinky:        return "PAUSE"
    if index and middle and ring and not pinky:            return "THREE"
    return "UNKNOWN"

# ── Drawing Engine ──
def draw_stroke(x, y, new_stroke=False):
    global last_x, last_y
    if new_stroke:
        last_x, last_y = x, y
        return

    cam_h, cam_w = canvas.shape[:2]
    alpha = brush_opacity

    if brush_type == "eraser":
        cv2.circle(canvas, (x, y), brush_size * 3, (0, 0, 0), -1)
        last_x, last_y = x, y
        return

    overlay = canvas.copy()

    if brush_type == "pen":
        cv2.line(overlay, (last_x, last_y), (x, y),
                 brush_color, brush_size, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    elif brush_type == "marker":
        cv2.line(overlay, (last_x, last_y), (x, y),
                 brush_color, brush_size * 3, cv2.LINE_AA)
        a = min(alpha, 0.45)
        cv2.addWeighted(overlay, a, canvas, 1 - a, 0, canvas)

    elif brush_type == "spray":
        radius  = brush_size * 3
        for _ in range(50):
            angle = np.random.uniform(0, 2 * math.pi)
            r     = np.random.uniform(0, radius)
            px    = int(x + r * math.cos(angle))
            py    = int(y + r * math.sin(angle))
            if 0 <= px < cam_w and 0 <= py < cam_h:
                cv2.circle(overlay, (px, py), 1, brush_color, -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    elif brush_type == "glow":
        glow = np.zeros_like(canvas)
        cv2.line(glow, (last_x, last_y), (x, y), brush_color, brush_size * 4, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), max(1.0, brush_size * 1.5))
        cv2.line(overlay, (last_x, last_y), (x, y),
                 brush_color, max(1, brush_size // 2), cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        cv2.addWeighted(glow, 0.4, canvas, 1.0, 0, canvas)

    elif brush_type == "neon":
        for w, a in [(brush_size * 5, 0.04), (brush_size * 2, 0.18),
                     (brush_size, 0.75), (max(1, brush_size // 3), 1.0)]:
            layer = canvas.copy()
            cv2.line(layer, (last_x, last_y), (x, y), brush_color, int(w), cv2.LINE_AA)
            cv2.addWeighted(layer, a * alpha, canvas, 1 - a * alpha, 0, canvas)

    last_x, last_y = x, y

# ── Panel helpers ──
def draw_section_label(panel, text, y):
    """Draw a section header with underline. Returns new y."""
    cv2.putText(panel, text, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 88, 130), 1, cv2.LINE_AA)
    cv2.line(panel, (12, y + 5), (PANEL_W - 12, y + 5), (40, 38, 62), 1)
    return y + 18

# ── Main Panel Draw ──
def draw_panel(frame, gesture, cam_h):
    panel = np.zeros((cam_h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (14, 13, 20)

    y = 12

    # ── LOGO ──────────────────────────────────────────
    cv2.putText(panel, "Air", (12, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Write", (52, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (108, 142, 255), 2, cv2.LINE_AA)
    cv2.circle(panel, (PANEL_W - 18, y + 12), 5, (108, 142, 255), -1, cv2.LINE_AA)
    y += 30
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (44, 42, 66), 1)
    y += 10

    # ── BRUSH TYPE ────────────────────────────────────
    y = draw_section_label(panel, "BRUSH TYPE", y)
    BTN_W, BTN_H, BTN_GAP = 100, 24, 5
    DOT_COLORS = {
        "pen":    (180, 210, 255), "marker": (108, 255, 180),
        "spray":  (255, 200, 108), "glow":   (255, 160, 255),
        "neon":   (80,  255, 120), "eraser": (170, 170, 200),
    }
    for i, bt in enumerate(BRUSH_TYPES):
        col = 12 + (i % 2) * (BTN_W + BTN_GAP)
        row = y + (i // 2) * (BTN_H + BTN_GAP)
        active = (brush_type == bt)
        fill   = (38, 34, 72) if active else (22, 20, 34)
        border = (108, 142, 255) if active else (48, 46, 68)
        tc     = (190, 210, 255) if active else (110, 108, 145)
        cv2.rectangle(panel, (col, row), (col + BTN_W, row + BTN_H), fill, -1)
        cv2.rectangle(panel, (col, row), (col + BTN_W, row + BTN_H), border, 1)
        dc = DOT_COLORS.get(bt, tc)
        cv2.circle(panel, (col + 11, row + BTN_H // 2), 4, dc, -1, cv2.LINE_AA)
        cv2.putText(panel, bt.capitalize(), (col + 20, row + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1, cv2.LINE_AA)
    y += 3 * (BTN_H + BTN_GAP) + 6

    # ── COLOR PALETTE ─────────────────────────────────
    y = draw_section_label(panel, "COLOR", y)
    # 5 per row, evenly spaced across panel width
    N_COLS   = 5
    DOT_R    = 11
    spacing  = (PANEL_W - 24) // N_COLS
    for i, c in enumerate(COLOR_CYCLE):
        col_i = i % N_COLS
        row_i = i // N_COLS
        cx = 12 + DOT_R + col_i * spacing
        cy = y + DOT_R + row_i * (DOT_R * 2 + 6)
        active = (c == brush_color)
        if active:
            cv2.circle(panel, (cx, cy), DOT_R + 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), DOT_R, c, -1, cv2.LINE_AA)
    rows_used = math.ceil(len(COLOR_CYCLE) / N_COLS)
    y += rows_used * (DOT_R * 2 + 6) + 8

    # ── BRUSH SIZE ────────────────────────────────────
    y = draw_section_label(panel, f"BRUSH SIZE  [ {brush_size} ]", y)
    bx1, bx2 = 12, PANEL_W - 12
    bmid = y + 3
    cv2.rectangle(panel, (bx1, bmid - 2), (bx2, bmid + 2), (40, 38, 60), -1)
    fill_x = bx1 + int((bx2 - bx1) * brush_size / 60)
    cv2.rectangle(panel, (bx1, bmid - 2), (fill_x, bmid + 2), (108, 142, 255), -1)
    pr = max(3, min(brush_size // 2, 10))
    cv2.circle(panel, (fill_x, bmid), pr, (108, 142, 255), -1, cv2.LINE_AA)
    y += 18

    # ── GESTURE GUIDE ─────────────────────────────────
    y = draw_section_label(panel, "GESTURE GUIDE", y)

    GESTURE_ROWS = [
        ("INDEX ONLY",   "Draw",       (108, 200, 255), "DRAW"),
        ("INDEX+MIDDLE", "Pause",      (100, 220, 255), "PAUSE"),
        ("OPEN PALM",    "Erase",      (80,  80,  240), "PALM"),
        ("THUMBS UP",    "Next Color", (200, 80,  255), "THUMB_UP"),
        ("ROCK ON",      "Save",       (80,  255, 140), "ROCK_ON"),
        ("SHAKA",        "Clear",      (255, 100,  80), "SHAKA"),
        ("3 FINGERS",    "Undo",       (80,  255, 220), "THREE"),
    ]
    ROW_H = 24
    for gname, gaction, gcolor, gkey in GESTURE_ROWS:
        active = (gesture == gkey)
        row_bg = (32, 28, 56) if active else (18, 16, 30)
        cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + ROW_H - 1), row_bg, -1)
        bar_color = gcolor if active else tuple(c // 3 for c in gcolor)
        cv2.rectangle(panel, (10, y), (13, y + ROW_H - 1), bar_color, -1)
        if active:
            cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + ROW_H - 1), gcolor, 1)

        name_c = (220, 218, 255) if active else (100, 98, 138)
        cv2.putText(panel, gname, (18, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, name_c, 1, cv2.LINE_AA)

        act_c = gcolor if active else tuple(c // 2 for c in gcolor)
        (tw, _), _ = cv2.getTextSize(gaction, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(panel, gaction, (PANEL_W - 12 - tw, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, act_c, 1, cv2.LINE_AA)
        y += ROW_H

    y += 6

    # ── KEYBOARD SHORTCUTS ────────────────────────────
    if y + 10 < cam_h - 55:
        y = draw_section_label(panel, "SHORTCUTS", y)
        SHORTCUTS = [
            ("[B]", "Next brush"),   ("[C]", "Next color"),
            ("[+]", "Size up"),      ("[-]", "Size down"),
            ("[U]", "Undo"),         ("[S]", "Save image"),   # ← updated label
            ("[Q]", "Quit"),
        ]
        for ks, desc in SHORTCUTS:
            if y + 12 > cam_h - 46:
                break
            cv2.putText(panel, ks, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (108, 142, 255), 1, cv2.LINE_AA)
            (kw, _), _ = cv2.getTextSize(ks, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.putText(panel, desc, (12 + kw + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (105, 102, 145), 1, cv2.LINE_AA)
            y += 15

    # ── TOAST ─────────────────────────────────────────
    if time.time() < toast_until:
        th = 26
        ty = cam_h - th - 6
        cv2.rectangle(panel, (8, ty), (PANEL_W - 8, ty + th), (24, 22, 50), -1)
        cv2.rectangle(panel, (8, ty), (PANEL_W - 8, ty + th), (100, 80, 200), 1)
        (tw, _), _ = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        tx = max(10, (PANEL_W - tw) // 2)
        cv2.putText(panel, toast_msg, (tx, ty + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 170, 255), 1, cv2.LINE_AA)

    # ── RIGHT BORDER ──────────────────────────────────
    cv2.line(panel, (PANEL_W - 1, 0), (PANEL_W - 1, cam_h), (50, 46, 75), 1)

    frame[:, :PANEL_W] = panel


# ── Main Loop ──
def main():
    global brush_type, brush_color, brush_size, is_drawing
    global last_x, last_y, brush_idx, color_index, canvas

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No camera found.")
        return

    # Auto-detect best resolution
    for w, h in [(1280, 720), (960, 540), (640, 480), (320, 240)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, test = cap.read()
        if ret and test is not None:
            print(f"[CAMERA] Running at {w}x{h}")
            break
    else:
        print("ERROR: Cannot read camera frames.")
        cap.release()
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

    show_toast("Ready! Show your hand.")

    print("\n" + "="*52)
    print("  AirWrite  —  Hand Gesture Drawing")
    print("="*52)
    print("  GESTURES")
    print("  Index finger only   ->  Draw")
    print("  Index + Middle      ->  Pause / lift pen")
    print("  Open palm           ->  Erase")
    print("  Thumbs up           ->  Next Color")
    print("  Rock On (index+pinky) -> Save image")
    print("  Shaka (thumb+pinky) ->  Clear canvas")
    print("  Pinch               ->  Next Color")
    print("  3 fingers           ->  Undo")
    print("\n  KEYBOARD: B=brush  C=color  +/-=size")
    print("            U=undo  S=save  Q=quit")       # ← save = S key only
    print("="*52 + "\n")

    current_gesture = "UNKNOWN"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_gesture = "UNKNOWN"

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]

            # Draw skeleton overlay
            mp_drawing.draw_landmarks(
                frame, lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            current_gesture = detect_gesture(lm)

            raw_x = int(lm.landmark[8].x * cam_w)
            raw_y = int(lm.landmark[8].y * cam_h)

            now = time.time()
            def oneshot(key, fn, delay=1.2):
                if now - gesture_debounce.get(key, 0) > delay:
                    fn()
                    gesture_debounce[key] = now

            if current_gesture == "DRAW":
                if not is_drawing:
                    save_snapshot()
                    draw_stroke(raw_x, raw_y, new_stroke=True)
                    is_drawing = True
                else:
                    draw_stroke(raw_x, raw_y)
                cv2.circle(frame, (raw_x, raw_y), brush_size + 4,
                           brush_color, 2, cv2.LINE_AA)

            elif current_gesture == "PALM":
                is_drawing = False
                prev = brush_type
                brush_type = "eraser"
                draw_stroke(raw_x, raw_y)
                brush_type = prev
                cv2.circle(frame, (raw_x, raw_y), brush_size * 3 + 4,
                           (80, 80, 255), 2, cv2.LINE_AA)

            elif current_gesture == "PAUSE":
                is_drawing = False
                cv2.circle(frame, (raw_x, raw_y), 14, (100, 220, 255), 2, cv2.LINE_AA)

            # ── CHANGED: THUMB_UP now cycles color instead of saving ──
            elif current_gesture == "THUMB_UP":
                is_drawing = False
                def next_color_thumb():
                    global color_index, brush_color
                    color_index = (color_index + 1) % len(COLOR_CYCLE)
                    brush_color = COLOR_CYCLE[color_index]
                    show_toast("Color changed")
                oneshot("thumb_color", next_color_thumb, delay=0.8)

            # ── Rock On (🤘) → Save ──
            elif current_gesture == "ROCK_ON":
                is_drawing = False
                oneshot("save", save_image, delay=1.5)

            # ── Shaka (🤙) → Clear canvas ──
            elif current_gesture == "SHAKA":
                is_drawing = False
                oneshot("clear", clear_canvas, delay=2.0)

            elif current_gesture == "THREE":
                is_drawing = False
                oneshot("undo", undo, delay=1.0)

            else:
                is_drawing = False
        else:
            is_drawing = False

        # ── Compose: blend canvas onto camera ──
        display = frame.copy()
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(display, display, mask=mask_inv)
        fg = cv2.bitwise_and(canvas,  canvas,  mask=mask)
        display = cv2.add(bg, fg)

        # ── Draw panel ──
        draw_panel(display, current_gesture, cam_h)

        cv2.imshow("AirWrite  |  Hand Gesture Drawing", display)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            brush_idx  = (brush_idx + 1) % len(BRUSH_TYPES)
            brush_type = BRUSH_TYPES[brush_idx]
            show_toast(f"Brush: {brush_type.capitalize()}")
        elif key == ord('c'):
            color_index = (color_index + 1) % len(COLOR_CYCLE)
            brush_color = COLOR_CYCLE[color_index]
            show_toast("Color changed")
        elif key in (ord('+'), ord('=')):
            brush_size = min(60, brush_size + 2)
            show_toast(f"Size: {brush_size}")
        elif key == ord('-'):
            brush_size = max(1, brush_size - 2)
            show_toast(f"Size: {brush_size}")
        elif key == ord('u'):
            undo()
        elif key in (8, 127, ord('d')):
            clear_canvas()
        elif key == ord('s'):         
            save_image()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()