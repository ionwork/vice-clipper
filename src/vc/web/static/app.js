// Live events (SSE), keys 1-9 = "Clip jetzt", cam rectangle editor.
(() => {
  const ev = document.getElementById("events");
  if (ev) {
    const es = new EventSource("/events");
    const fmt = (m) => {
      const t = (m.ts ? new Date(m.ts) : new Date()).toLocaleTimeString("de-DE");
      const who = m.clip_id || m.streamer_id || "";
      const extra = m.kind === "trigger" ? `z ${m.z ?? ""} pre ${m.pre_score ?? ""}`
        : m.kind === "judged" ? `hype ${m.hype} ${m.keep ? "KEEP" : "weg"} ${m.reason || ""}`
        : m.error || m.reason || "";
      return `${t} ${m.kind} ${who} ${extra}`;
    };
    es.onmessage = (e) => {
      const m = JSON.parse(e.data);
      const d = document.createElement("div");
      d.className = "k-" + m.kind;
      d.textContent = fmt(m);
      ev.prepend(d);
      while (ev.childElementCount > 300) ev.lastChild.remove();
      if (["rendered", "judged", "candidate", "delivered"].includes(m.kind)) htmx.trigger("#clips", "refresh");
    };
    es.addEventListener("tick", (e) => {
      for (const [sid, s] of Object.entries(JSON.parse(e.data))) {
        const el = document.querySelector(`[data-z="${sid}"]`);
        if (el) el.textContent = Number(s.z).toFixed(1);
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (!/^[1-9]$/.test(e.key) || e.target.closest("input,select,textarea")) return;
    const tile = document.querySelectorAll("#tiles .tile")[Number(e.key) - 1];
    if (tile) fetch(`/api/clip/${tile.dataset.sid}`, { method: "POST" })
      .then((r) => r.json()).then((j) => console.log("clip", j));
  });

  // cam editor: drag on a 16:9 stage, saved in 1920x1080 coordinates
  for (const box of document.querySelectorAll(".cam-editor")) {
    const stage = box.querySelector(".stage"), rect = box.querySelector(".rect"), out = box.querySelector(".out");
    let cam = JSON.parse(box.dataset.cam || "null"), start = null;
    const scale = () => 1920 / stage.clientWidth;
    const show = () => {
      if (!cam) { rect.style.display = "none"; return; }
      const s = scale();
      Object.assign(rect.style, { display: "block", left: cam.x / s + "px", top: cam.y / s + "px", width: cam.w / s + "px", height: cam.h / s + "px" });
      out.textContent = `x ${cam.x}, y ${cam.y}, w ${cam.w}, h ${cam.h}`;
    };
    const pos = (e) => { const r = stage.getBoundingClientRect(); return [Math.max(0, Math.min(r.width, e.clientX - r.left)), Math.max(0, Math.min(r.height, e.clientY - r.top))]; };
    stage.addEventListener("mousedown", (e) => { start = pos(e); });
    window.addEventListener("mousemove", (e) => {
      if (!start) return;
      const [x, y] = pos(e), s = scale();
      cam = { x: Math.round(Math.min(x, start[0]) * s), y: Math.round(Math.min(y, start[1]) * s), w: Math.round(Math.abs(x - start[0]) * s), h: Math.round(Math.abs(y - start[1]) * s) };
      show();
    });
    window.addEventListener("mouseup", () => { start = null; });
    const post = (body) => fetch(`/api/streamers/${box.dataset.sid}/cam`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => { out.textContent = r.ok ? "gespeichert ✓" : "Fehler " + r.status; });
    box.querySelector(".save").onclick = () => cam && cam.w > 20 && cam.h > 20 ? post(cam) : (out.textContent = "erst Rahmen ziehen");
    box.querySelector(".clear").onclick = () => { cam = null; show(); post(null); };
    box.querySelector(".suggest").onclick = () => {
      out.textContent = "suche Gesicht …";
      fetch(`/api/streamers/${box.dataset.sid}/suggest`, { method: "POST" }).then((r) => r.json())
        .then((j) => { if (j.rect) { cam = j.rect; show(); out.textContent += " (Vorschlag, noch speichern)"; } else out.textContent = j.detail || "kein Gesicht gefunden"; });
    };
    window.addEventListener("resize", show);
    show();
  }
})();
