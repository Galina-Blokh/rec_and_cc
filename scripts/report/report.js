(function () {
  "use strict";
  var root = document.documentElement;
  try { var saved = localStorage.getItem("rcc-report-theme"); if (saved) root.setAttribute("data-theme", saved); } catch (err) { /* storage blocked */ }
  var btn = document.getElementById("themeBtn");
  if (btn) btn.addEventListener("click", function () {
    var dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("rcc-report-theme", next); } catch (err) { /* storage blocked */ }
  });
  document.querySelectorAll("[data-tv]").forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.closest("figure").querySelector(".tview");
      v.classList.toggle("on");
      b.textContent = v.classList.contains("on") ? "Hide table" : "Table";
    });
  });
  var tip = document.getElementById("tip");
  document.addEventListener("mousemove", function (ev) {
    var t = ev.target.closest && ev.target.closest("[data-tip]");
    if (!t) { tip.classList.remove("on"); return; }
    tip.textContent = t.getAttribute("data-tip");
    var x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px"; tip.classList.add("on");
  });
  document.querySelectorAll("table.sortable").forEach(function (table) {
    table.querySelectorAll("th").forEach(function (th, col) {
      th.addEventListener("click", function () {
        var asc = !th.classList.contains("asc");
        table.querySelectorAll("th").forEach(function (h) { h.classList.remove("asc", "desc"); });
        th.classList.add(asc ? "asc" : "desc");
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        var key = function (r) {
          var s = r.cells[col].textContent.trim();
          var num = parseFloat(s.replace(/[#,$%]/g, ""));
          return isNaN(num) ? s.toLowerCase() : num;
        };
        rows.sort(function (a, b) { var x = key(a), y = key(b); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });
  document.querySelectorAll("[data-filter]").forEach(function (input) {
    input.addEventListener("input", function () {
      var q = input.value.toLowerCase();
      document.querySelectorAll("." + input.getAttribute("data-filter") + " tbody tr").forEach(function (r) {
        r.style.display = r.textContent.toLowerCase().indexOf(q) >= 0 ? "" : "none";
      });
    });
  });
})();
