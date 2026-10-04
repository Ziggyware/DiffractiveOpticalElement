# DOE — the picture is not in here
1. `1…6` or the chips pick a scene; `tour` cycles one scene per 10 s (every animated quantity shares that 10 s clock, so the piece loops).
2. Scroll / pinch the **left** panel to zoom the mask 1× → 64× (past single pixels), drag to pan; scroll / pinch the **right** panel to zoom the replay.
3. Drag the divider to re-split the view; `space` = calm (freezes all motion, also honours `prefers-reduced-motion`), `r` = record 10 s, `0` = reset the view.
4. Every setting (scene, λ, pitch, levels, text, seed …) rides in the URL fragment, ≤200 chars - copy the link to reproduce the exact same frame.
5. `?selftest` in the URL (or the selftest button) runs the graded physics checks on screen and in the console; `DOE.report()` dumps the live state.
