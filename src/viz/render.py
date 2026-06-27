"""Per-step pygame visualisation of a matchup (CC3 has no built-in renderer).

    python render.py <run_id|dir>            # interactive: SPACE play/pause, <-/-> step, R reset, ESC quit
    python render.py <run_id|dir> --gif      # headless, saves figs/animation.gif (per-step animation)
    python render.py <run_id|dir> --smoke    # headless, saves figs/pygame_preview.png (single frame)

`<run_id|dir>` may be a results id (e.g. sweep_..._abc/rule_vs_rule) or a path.
Team by FILL colour: blue=friendly, red=compromised. Platform by SHAPE:
triangle=UAV, square=UGV. Effects: purple ring=jammed, orange arrow=GPS spoof
(true->reported, clamped to the grid), yellow ring=detected by blue.

sweep.py imports save_gif() to write one animation per matchup.
"""
import os, sys, json
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

_FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
if _FLAGS & {"--gif", "--smoke"} or os.environ.get("SDL_VIDEODRIVER"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # headless
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame

MAP, PANEL, PAD = 720, 270, 20
BG, GRID_C, FG = (18, 20, 28), (40, 44, 56), (230, 230, 235)
FRIENDLY, RED = (70, 130, 235), (225, 70, 55)              # team fill: friendly / compromised
JAM, SPOOF, DET = (180, 90, 220), (255, 165, 0), (255, 230, 60)   # effects: jam / spoof / detect


def _resolve(arg):
    return arg if os.path.isdir(arg) else os.path.join(RESULTS, arg)


def _load(d):
    z = np.load(os.path.join(d, "arrays.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    return {"types": list(z["types"]), "grid": meta["config"]["fleet"]["grid"],
            "pt": z["pos_true"][0], "pr": z["pos_rep"][0], "red": z["red_owned"][0],
            "ljam": z["label_jam"][0], "lgps": z["label_gps"][0],
            "dj": z["det_jam"][0], "dg": z["det_gps"][0],
            "dc": z["det_comp"][0] if "det_comp" in z.files else z["det_jam"][0] * 0,
            "meta": meta}


def _screen(data):
    pygame.init()
    screen = pygame.display.set_mode((MAP + PANEL, MAP))
    pygame.display.set_caption(f"DroneSwarm — {data['meta']['config']['name']}")
    return screen, pygame.font.SysFont("consolas", 16), pygame.font.SysFont("consolas", 20, bold=True)


def _draw(screen, data, t, font, big):
    grid, types, n = data["grid"], data["types"], len(data["types"])
    pt, pr, red = data["pt"], data["pr"], data["red"]
    ljam, lgps, dj, dg, dc = data["ljam"], data["lgps"], data["dj"], data["dg"], data["dc"]
    meta = data["meta"]; T = pt.shape[0]
    scale = (MAP - 2 * PAD) / grid
    sp = lambda p: (int(PAD + p[0] * scale), int(MAP - PAD - p[1] * scale))

    screen.fill(BG)
    for g in range(0, int(grid) + 1, 20):
        pygame.draw.line(screen, GRID_C, sp((g, 0)), sp((g, grid)))
        pygame.draw.line(screen, GRID_C, sp((0, g)), sp((grid, g)))

    counts = {"compromised": 0, "jammed": 0, "spoofed": 0}
    for e in range(n):
        p = sp(pt[t, e])
        team = RED if red[t, e] else FRIENDLY                # fill = faction (team)
        if lgps[t, e]:                                        # spoof arrow, clamped to grid
            rep = sp(np.clip(pr[t, e], 0, grid))
            pygame.draw.line(screen, SPOOF, p, rep, 2)
            pygame.draw.circle(screen, SPOOF, rep, 5, 1)
            counts["spoofed"] += 1
        if types[e] == "uav":                                # shape = platform
            pygame.draw.polygon(screen, team, [(p[0], p[1] - 9), (p[0] - 9, p[1] + 9), (p[0] + 9, p[1] + 9)])
        else:
            pygame.draw.rect(screen, team, (p[0] - 9, p[1] - 9, 18, 18))
        if ljam[t, e]:
            pygame.draw.circle(screen, JAM, p, 15, 2); counts["jammed"] += 1
        if dj[t, e] or dg[t, e] or dc[t, e]:
            pygame.draw.circle(screen, DET, p, 19, 1)
        counts["compromised"] += int(red[t, e])
        screen.blit(font.render(str(e), True, FG), (p[0] + 6, p[1] - 8))

    x = MAP + 14
    screen.blit(big.render(f"step {t}/{T - 1}", True, FG), (x, 18))
    dfn = meta.get("defense", {})
    info = [("scenario", meta["config"]["name"]),
            ("red", meta.get("red_type", "?")), ("blue", meta.get("blue_type", "?")),
            ("defense", f"{dfn.get('detector', 'none')}/{dfn.get('response', 'none')}"), ("", ""),
            *[(k, str(v)) for k, v in counts.items()], ("", ""),
            ("comp F1", str(dfn.get("comp_F1", "-"))),
            ("jam F1", str(dfn.get("jam_F1", "-"))), ("gps F1", str(dfn.get("gps_F1", "-")))]
    for i, (k, v) in enumerate(info):
        screen.blit(font.render(f"{k:13}{v}" if k else "", True, FG), (x, 56 + i * 24))
    y0 = 56 + len(info) * 24 + 12
    glyph = (155, 158, 168)                                   # faction-neutral shape glyphs
    pygame.draw.polygon(screen, glyph, [(x + 7, y0), (x + 1, y0 + 12), (x + 13, y0 + 12)])
    screen.blit(font.render("UAV (triangle)", True, FG), (x + 22, y0))
    pygame.draw.rect(screen, glyph, (x + 1, y0 + 24, 12, 12))
    screen.blit(font.render("UGV (square)", True, FG), (x + 22, y0 + 24))
    for i, (txt, col, ring) in enumerate([("friendly", FRIENDLY, False), ("compromised", RED, False),
                                          ("jammed", JAM, True), ("gps spoof", SPOOF, False),
                                          ("detected", DET, True)]):
        y = y0 + 48 + i * 22
        pygame.draw.circle(screen, col, (x + 7, y + 7), 7, 2 if ring else 0)
        screen.blit(font.render(txt, True, FG), (x + 22, y))
    screen.blit(font.render("SPACE <- -> R ESC", True, (140, 140, 150)), (x, MAP - 30))
    return counts


def interactive(d):
    data = _load(d); T = data["pt"].shape[0]
    screen, font, big = _screen(data)
    clock = pygame.time.Clock(); t, playing = 0, True
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE: playing = not playing
                elif ev.key == pygame.K_RIGHT: t, playing = min(T - 1, t + 1), False
                elif ev.key == pygame.K_LEFT: t, playing = max(0, t - 1), False
                elif ev.key == pygame.K_r: t = 0
        _draw(screen, data, t, font, big)
        pygame.display.flip()
        if playing:
            t = (t + 1) % T
        clock.tick(6)


def save_preview(d):
    data = _load(d); screen, font, big = _screen(data)
    _draw(screen, data, data["pt"].shape[0] // 2, font, big)
    figs = os.path.join(d, "figs"); os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, "pygame_preview.png")
    pygame.image.save(screen, out); pygame.quit()
    print(f"preview -> {out}")
    return out


def save_gif(d, scale=0.5, fps=6):
    """Render the whole episode headless into figs/animation.gif (per-step animation)."""
    from PIL import Image
    data = _load(d); T = data["pt"].shape[0]
    screen, font, big = _screen(data)
    frames = []
    for t in range(T):
        _draw(screen, data, t, font, big)
        img = Image.frombytes("RGB", screen.get_size(), pygame.image.tostring(screen, "RGB"))
        if scale != 1:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        frames.append(img)
    pygame.quit()
    figs = os.path.join(d, "figs"); os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, "animation.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"gif -> {out}")
    return out


if __name__ == "__main__":
    d = _resolve([a for a in sys.argv[1:] if not a.startswith("--")][0])
    if "--gif" in _FLAGS:
        save_gif(d)
    elif "--smoke" in _FLAGS:
        save_preview(d)
    else:
        interactive(d)
