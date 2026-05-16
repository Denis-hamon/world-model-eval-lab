/*
 * Vanilla JS that powers five pieces of interactivity on the live site:
 *
 *   1. Theme toggle (light / dark). The theme is bootstrapped synchronously
 *      from <head> in default.html to avoid a white flash on dark-mode load;
 *      this file only handles the click-to-toggle and the localStorage write.
 *   2. Mobile-friendly nav toggle (collapses the section links into a
 *      drop-down below 880px).
 *   3. Auto-built right-rail table of contents from the page's h2/h3 nodes,
 *      with a scroll-spy that highlights the section currently in view.
 *   4. Scroll-triggered reveal animations (IntersectionObserver adding
 *      .is-visible to .reveal and to a couple of named figures).
 *   5. Hover tooltips on the inline horizon-sweep SVG, reading per-horizon
 *      values from data-* attributes on each `.data-slice` group.
 *
 * No frameworks, no build step.
 */

(function () {
  "use strict";

  function reduceMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  // --- 1. Theme toggle ------------------------------------------------------

  function setupThemeToggle() {
    var btn = document.querySelector("[data-theme-toggle]");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme") || "light";
      var next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("wmel-theme", next); } catch (e) { /* ignore */ }
      btn.setAttribute("aria-label", next === "dark" ? "Switch to light mode" : "Switch to dark mode");
    });
    var initial = document.documentElement.getAttribute("data-theme") || "light";
    btn.setAttribute("aria-label", initial === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }

  // --- 2. Mobile nav toggle -------------------------------------------------

  function setupNavToggle() {
    var btn = document.querySelector("[data-nav-toggle]");
    var nav = document.querySelector(".site-nav");
    if (!btn || !nav) return;
    btn.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll(".site-nav-links a").forEach(function (a) {
      a.addEventListener("click", function () {
        nav.classList.remove("is-open");
        btn.setAttribute("aria-expanded", "false");
      });
    });
  }

  // --- 3. Right-rail TOC + scroll-spy --------------------------------------

  function slugify(text) {
    return text.toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "");
  }

  function setupToc() {
    var tocList = document.querySelector("[data-toc]");
    if (!tocList) return;
    var main = document.querySelector(".main-content");
    if (!main) return;
    var EXCLUDE_TOC_SCOPES = ".policy-card, .release-card, .path-card, .verdict-legend, .whats-new, .release-banner, header";
    var allHeadings = main.querySelectorAll("h2, h3");
    var headings = Array.prototype.filter.call(allHeadings, function (h) {
      return !h.closest(EXCLUDE_TOC_SCOPES);
    });
    if (headings.length < 2) {
      var aside = tocList.closest(".site-toc");
      if (aside) aside.style.display = "none";
      return;
    }
    var items = [];
    headings.forEach(function (h) {
      if (!h.id) h.id = slugify(h.textContent || "");
      if (!h.id) return;
      var li = document.createElement("li");
      li.className = h.tagName === "H3" ? "toc-h3" : "toc-h2";
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent || "";
      li.appendChild(a);
      tocList.appendChild(li);
      items.push({ heading: h, link: a });
    });
    if (!("IntersectionObserver" in window)) return;
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var id = entry.target.id;
        var match = items.find(function (it) { return it.heading.id === id; });
        if (!match) return;
        if (entry.isIntersecting) {
          items.forEach(function (it) { it.link.classList.remove("is-active"); });
          match.link.classList.add("is-active");
        }
      });
    }, { rootMargin: "-30% 0px -60% 0px", threshold: 0 });
    items.forEach(function (it) { spy.observe(it.heading); });
  }

  // --- 4. Scroll reveals ----------------------------------------------------

  function setupReveals() {
    var targets = document.querySelectorAll(".reveal, .chart-container, img.figure-architecture-img");
    if (!("IntersectionObserver" in window) || reduceMotion()) {
      targets.forEach(function (el) { el.classList.add("is-visible", "is-revealed"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible", "is-revealed");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
    targets.forEach(function (el) { io.observe(el); });
  }

  // --- 5. Sweep chart tooltips ---------------------------------------------

  function fmt(value) {
    return (value === undefined || value === null) ? "n/a" : value;
  }

  function setupSweepTooltips() {
    var containers = document.querySelectorAll(".chart-container.has-tooltips");
    containers.forEach(function (container) {
      var svg = container.querySelector("svg.figure-sweep");
      if (!svg) return;

      var tooltip = document.createElement("div");
      tooltip.className = "sweep-tooltip";
      tooltip.setAttribute("role", "status");
      tooltip.setAttribute("aria-live", "polite");
      container.appendChild(tooltip);

      var slices = svg.querySelectorAll(".data-slice");
      slices.forEach(function (slice) {
        slice.addEventListener("mouseenter", function () { showTooltip(container, svg, tooltip, slice); });
        slice.addEventListener("focusin",   function () { showTooltip(container, svg, tooltip, slice); });
        slice.addEventListener("mouseleave", function () { hideTooltip(tooltip, slice); });
        slice.addEventListener("focusout",   function () { hideTooltip(tooltip, slice); });
        slice.setAttribute("tabindex", "0");
      });
    });
  }

  function showTooltip(container, svg, tooltip, slice) {
    slice.classList.add("is-active");

    var horizon = slice.getAttribute("data-h");
    var success = slice.getAttribute("data-success");
    var latency = slice.getAttribute("data-latency");
    var compute = slice.getAttribute("data-compute");
    var steps = slice.getAttribute("data-steps");

    tooltip.innerHTML =
      "<strong>plan_horizon = " + fmt(horizon) + "</strong>" +
      "<div class='row'><span class='k'>success rate</span><span class='v accent'>" + fmt(success) + "</span></div>" +
      "<div class='row'><span class='k'>latency / call</span><span class='v warn'>" + fmt(latency) + " ms</span></div>" +
      "<div class='row'><span class='k'>compute / decision</span><span class='v'>" + fmt(compute) + "</span></div>" +
      "<div class='row'><span class='k'>avg steps</span><span class='v'>" + fmt(steps) + "</span></div>";

    var successPoint = slice.querySelector(".success-point");
    if (!successPoint) return;
    var containerRect = container.getBoundingClientRect();
    var pointBox = successPoint.getBoundingClientRect();
    var x = pointBox.left + pointBox.width / 2 - containerRect.left;
    var y = pointBox.top - containerRect.top;
    tooltip.style.transform = "translate(calc(-50% + " + x.toFixed(1) + "px), calc(-100% + " + (y - 4).toFixed(1) + "px))";
    tooltip.classList.add("is-visible");
  }

  function hideTooltip(tooltip, slice) {
    slice.classList.remove("is-active");
    tooltip.classList.remove("is-visible");
  }

  // --- bootstrap ------------------------------------------------------------

  function boot() {
    setupThemeToggle();
    setupNavToggle();
    setupToc();
    setupReveals();
    setupSweepTooltips();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
