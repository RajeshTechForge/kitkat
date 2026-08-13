import { useState, useEffect } from "react";

type ThemeMode = "system" | "light" | "dark";

export default function ThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>("system");

  useEffect(() => {
    const saved = localStorage.getItem("doc-theme") as ThemeMode | null;
    if (saved && ["system", "light", "dark"].includes(saved)) {
      setMode(saved);
    }
  }, []);

  useEffect(() => {
    const applyTheme = (currentMode: ThemeMode) => {
      let activeTheme: "light" | "dark" = "dark";
      if (currentMode === "system") {
        activeTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
      } else {
        activeTheme = currentMode;
      }

      document.documentElement.dataset.docTheme = activeTheme;
      document.documentElement.dataset.docThemeMode = currentMode;
      localStorage.setItem("doc-theme", currentMode);
    };

    applyTheme(mode);

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      if (mode === "system") {
        applyTheme("system");
      }
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [mode]);

  const cycleTheme = () => {
    setMode((prev) => {
      if (prev === "system") return "light";
      if (prev === "light") return "dark";
      return "system";
    });
  };

  const getLabel = () => {
    if (mode === "system") return "System";
    if (mode === "light") return "Light";
    return "Dark";
  };

  return (
    <button
      type="button"
      onClick={cycleTheme}
      className="pane tab flex items-center gap-1.5 px-3 h-full text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors cursor-pointer"
      title={`Theme: ${getLabel()} (Click to switch)`}
      aria-label={`Current theme ${getLabel()}. Click to switch theme.`}
    >
      {mode === "system" && (
        <svg
          width="14"
          height="14"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.3"
          className="shrink-0"
        >
          <rect x="2" y="3" width="12" height="8" rx="1"></rect>
          <path d="M6 14h4M8 11v3"></path>
        </svg>
      )}

      {mode === "light" && (
        <svg
          width="14"
          height="14"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.3"
          className="shrink-0 text-amber-500"
        >
          <circle cx="8" cy="8" r="3"></circle>
          <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"></path>
        </svg>
      )}

      {mode === "dark" && (
        <svg
          width="14"
          height="14"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.3"
          className="shrink-0 text-sky-400"
        >
          <path d="M12.3 9.4A6 6 0 1 1 6.6 3.7a4.8 4.8 0 0 0 5.7 5.7z"></path>
        </svg>
      )}

      <span className="text-[12px] font-mono capitalize hidden sm:inline">
        {getLabel()}
      </span>
    </button>
  );
}
