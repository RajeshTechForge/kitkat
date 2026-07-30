import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

interface Entry {
  section: string;
  label: string;
  href: string;
}

const INDEX: Entry[] = [
  { section: "Docs", label: "Introduction", href: "/docs" },
  { section: "Docs", label: "Installation", href: "/docs/installation" },
  { section: "Docs", label: "Core Concepts", href: "/docs/core-concepts" },
  {
    section: "Docs",
    label: "API Reference — pipeline()",
    href: "/docs/api-reference",
  },
  { section: "Docs", label: "Examples & Tutorials", href: "/docs/examples" },
  { section: "Docs", label: "Contributing", href: "/docs/contributing" },
  { section: "Docs", label: "Changelog", href: "/docs/changelog" },
  { section: "Site", label: "Homepage", href: "/" },
  { section: "Site", label: "Features", href: "/#features" },
  {
    section: "Site",
    label: "GitHub Repository",
    href: "https://github.com/RajeshTechForge/kitkat",
  },
];

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return INDEX;
    return INDEX.filter(
      (e) =>
        e.label.toLowerCase().includes(q) ||
        e.section.toLowerCase().includes(q),
    );
  }, [query]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isMod = e.metaKey || e.ctrlKey;
      if (isMod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
  }, [open]);

  useEffect(() => setActiveIndex(0), [query]);

  function navigate(href: string) {
    setOpen(false);
    window.location.href = href;
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results[activeIndex]) {
      navigate(results[activeIndex].href);
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-label="Open search"
        className="pane tab flex items-center gap-3 px-4 h-full w-full text-left text-text-tertiary hover:text-text-secondary transition-colors"
      >
        <svg
          width="13"
          height="13"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
        >
          <circle cx="7" cy="7" r="5.25" />
          <path d="M11 11L15 15" />
        </svg>
        <span className="flex-1 hidden sm:inline">Search docs…</span>
        <span className="kbd hidden sm:inline">⌘K</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4"
          style={{ background: "rgba(10, 9, 8, 0.7)" }}
          onClick={() => setOpen(false)}
        >
          <div
            className="grid-frame grid-rows-[auto_1fr] w-full max-w-xl max-h-[60vh] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="pane pane-raised flex items-center gap-3 px-4 h-12">
              <svg
                width="14"
                height="14"
                viewBox="0 0 16 16"
                fill="none"
                stroke="#968d83"
                strokeWidth="1.4"
              >
                <circle cx="7" cy="7" r="5.25" />
                <path d="M11 11L15 15" />
              </svg>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Search introduction, installation, api…"
                className="flex-1 bg-transparent outline-none font-mono text-[13.5px] text-text-primary placeholder:text-text-tertiary"
              />
              <span className="kbd">esc</span>
            </div>
            <div className="pane overflow-y-auto">
              {results.length === 0 && (
                <p className="px-4 py-6 text-text-tertiary text-[13px] font-mono">
                  No results for "{query}"
                </p>
              )}
              {results.map((r, i) => (
                <button
                  key={r.href + r.label}
                  onClick={() => navigate(r.href)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`w-full text-left px-4 py-3 flex items-center justify-between border-b border-border last:border-b-0 transition-colors ${
                    i === activeIndex ? "bg-bg-tertiary" : "bg-bg-primary"
                  }`}
                >
                  <span className="flex items-center gap-3">
                    <span className="eyebrow w-14 shrink-0">{r.section}</span>
                    <span className="text-[13.5px] text-text-primary">
                      {r.label}
                    </span>
                  </span>
                  {i === activeIndex && (
                    <span className="text-accent text-[11px] font-mono">↵</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
