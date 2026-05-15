/*
 * Vanilla JS that powers two pieces of interactivity on the live site:
 *
 *   1. Scroll-triggered reveal animations (IntersectionObserver adding
 *      .is-visible to .reveal and to a couple of named figures).
 *   2. Hover tooltips on the inline horizon-sweep SVG, reading per-horizon
 *      values from data-* attributes on each `.data-slice` group.
 *
 * No frameworks, no build step.
 */

(function () {
  "use strict";

  function reduceMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  // --- 1. Scroll reveals ----------------------------------------------------

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

  // --- 2. Sweep chart tooltips ---------------------------------------------

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
        // Make the slice keyboard-focusable.
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

    // Position the tooltip above the success-point of the slice.
    var successPoint = slice.querySelector(".success-point");
    if (!successPoint) return;
    var svgRect = svg.getBoundingClientRect();
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      setupReveals();
      setupSweepTooltips();
    });
  } else {
    setupReveals();
    setupSweepTooltips();
  }
})();
