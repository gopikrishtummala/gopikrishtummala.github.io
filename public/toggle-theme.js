const primaryColorScheme = ""; // "light" | "dark" | "cyber" | "academic"

// Available themes in order
const themes = ["light", "dark", "cyber", "academic"];

// Get theme data from local storage
const currentTheme = localStorage.getItem("theme");

function getPreferTheme() {
  // return theme value in local storage if it is set
  if (currentTheme && themes.includes(currentTheme)) return currentTheme;

  // return primary color scheme if it is set
  if (primaryColorScheme && themes.includes(primaryColorScheme)) return primaryColorScheme;

  // return user device's prefer color scheme (map to light/dark)
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

let themeValue = getPreferTheme();

function setPreference() {
  localStorage.setItem("theme", themeValue);
  reflectPreference();
}

function reflectPreference() {
  document.firstElementChild.setAttribute("data-theme", themeValue);

  const themeBtn = document.querySelector("#theme-btn");
  const themeIndicator = document.querySelector("#theme-indicator");
  
  if (themeBtn) {
    const themeName = themeValue.charAt(0).toUpperCase() + themeValue.slice(1);
    themeBtn.setAttribute("aria-label", `Current theme: ${themeValue}. Click to cycle themes.`);
    themeBtn.setAttribute("title", `Theme: ${themeName} (Click to cycle)`);
  }
  
  // Update theme indicator
  if (themeIndicator) {
    const themeColors = {
      light: "#10b981",
      dark: "#f59e0b",
      cyber: "#a855f7",
      academic: "#0d9488"
    };
    themeIndicator.style.backgroundColor = themeColors[themeValue] || themeColors.light;
    themeIndicator.textContent = themeValue.charAt(0).toUpperCase();
    themeIndicator.classList.remove("opacity-0");
  }

  // Get a reference to the body element
  const body = document.body;

  // Check if the body element exists before using getComputedStyle
  if (body) {
    // Get the computed styles for the body element
    const computedStyles = window.getComputedStyle(body);

    // Get the background color property
    const bgColor = computedStyles.backgroundColor;

    // Set the background color in <meta theme-color ... />
    document
      .querySelector("meta[name='theme-color']")
      ?.setAttribute("content", bgColor);
  }
}

// set early so no page flashes / CSS is made aware
reflectPreference();

window.onload = () => {
  function setThemeFeature() {
    // set on load so screen readers can get the latest value on the button
    reflectPreference();

    // now this script can find and listen for clicks on the control
    document.querySelector("#theme-btn")?.addEventListener("click", () => {
      const currentIndex = themes.indexOf(themeValue);
      const nextIndex = (currentIndex + 1) % themes.length;
      themeValue = themes[nextIndex];
      setPreference();
    });
  }

  setThemeFeature();

  // Runs on view transitions navigation
  document.addEventListener("astro:after-swap", setThemeFeature);
};

// Set theme-color value before page transition
// to avoid navigation bar color flickering in Android dark mode
document.addEventListener("astro:before-swap", event => {
  const bgColor = document
    .querySelector("meta[name='theme-color']")
    ?.getAttribute("content");

  event.newDocument
    .querySelector("meta[name='theme-color']")
    ?.setAttribute("content", bgColor);
});

// sync with system changes
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", ({ matches: isDark }) => {
    themeValue = isDark ? "dark" : "light";
    setPreference();
  });
