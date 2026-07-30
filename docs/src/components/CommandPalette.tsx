import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import type { SearchItem } from "../utils/docs";

interface Props {
  items?: SearchItem[];
}

export default function CommandPalette({ items = [] }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        e.category.toLowerCase().includes(q) ||
        (e.snippet && e.snippet.toLowerCase().includes(q)),
    );
  }, [query, items]);

  useEffect(() => {
    function onKey(e: window.KeyboardEvent) {
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
        <span className="flex-1 hidden sm:inline text-[13px]">
          Search docs…
        </span>
        <span className="kbd hidden sm:inline">⌘K</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] sm:pt-[12vh] px-4"
          style={{
            background: "rgba(10, 9, 8, 0.75)",
            backdropFilter: "blur(2px)",
          }}
          onClick={() => setOpen(false)}
        >
          <div
            className="grid-frame grid-rows-[auto_1fr] w-full max-w-xl max-h-[70vh] shadow-2xl bg-bg-primary border border-border rounded-none"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="pane pane-raised flex items-center gap-3 px-4 h-12 border-b border-border">
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
                placeholder="Search documentation, API, concepts…"
                className="flex-1 bg-transparent outline-none font-mono text-[13.5px] text-text-primary placeholder:text-text-tertiary"
              />
              <span className="kbd">esc</span>
            </div>

            <div className="pane overflow-y-auto max-h-[calc(70vh-3rem)]">
              {results.length === 0 && (
                <p className="px-5 py-8 text-text-tertiary text-[13px] font-mono text-center">
                  No docs found for "{query}"
                </p>
              )}
              {results.map((r, i) => (
                <button
                  key={r.id + i}
                  onClick={() => navigate(r.href)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`w-full text-left px-5 py-3 flex items-start justify-between border-b border-border last:border-b-0 transition-colors ${
                    i === activeIndex ? "bg-bg-tertiary" : "bg-bg-primary"
                  }`}
                >
                  <div className="flex flex-col gap-0.5 min-w-0 pr-4">
                    <div className="flex items-center gap-2">
                      <span className="eyebrow text-[10px] text-text-tertiary shrink-0">
                        {r.category}
                      </span>
                      {r.type === "heading" && (
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-bg-tertiary text-text-tertiary border border-border">
                          Section
                        </span>
                      )}
                    </div>
                    <span className="text-[13.5px] font-medium text-text-primary truncate">
                      {r.title}
                    </span>
                    {r.snippet && (
                      <span className="text-[12px] text-text-tertiary truncate font-mono">
                        {r.snippet}
                      </span>
                    )}
                  </div>
                  {i === activeIndex && (
                    <span className="text-accent text-[11px] font-mono shrink-0 self-center">
                      ↵
                    </span>
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
