import { useState } from "react";
import type { NavGroup } from "../utils/docs";

interface Props {
  groups: NavGroup[];
  currentPath: string;
}

function normalizePath(path: string): string {
  try {
    const clean = decodeURIComponent(path.split("#")[0].replace(/\/$/, ""));
    return clean === "" ? "/docs" : clean;
  } catch {
    return path;
  }
}

function isPathActive(currentPath: string, href: string): boolean {
  const current = normalizePath(currentPath);
  const target = normalizePath(href);
  if (target === "/docs") return current === "/docs";
  return current === target;
}

export default function DocsSidebar({ groups, currentPath }: Props) {
  // Collapsible category state (all open by default)
  const [collapsedCategories, setCollapsedCategories] = useState<
    Record<string, boolean>
  >({});

  const toggleCategory = (cat: string) => {
    setCollapsedCategories((prev) => ({
      ...prev,
      [cat]: !prev[cat],
    }));
  };

  return (
    <nav className="flex flex-col py-2 text-[13.5px]">
      {groups.map((group) => {
        const isCollapsed = !!collapsedCategories[group.category];

        return (
          <div key={group.category} className="mb-4 last:mb-0">
            {/* Category Header */}
            <div className="flex items-center justify-between px-5 py-1.5 mb-1 group">
              <span className="eyebrow text-[11px] font-mono text-text-tertiary uppercase tracking-wider">
                {group.category}
              </span>
              <button
                type="button"
                aria-label={
                  isCollapsed
                    ? `Expand ${group.category}`
                    : `Collapse ${group.category}`
                }
                onClick={() => toggleCategory(group.category)}
                className="text-text-tertiary hover:text-text-primary transition-colors p-1 -mr-1"
              >
                <svg
                  width="8"
                  height="8"
                  viewBox="0 0 9 9"
                  className={`transition-transform duration-200 ${isCollapsed ? "-rotate-90" : ""}`}
                >
                  <path
                    d="M2 1L6.5 4.5L2 8"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    fill="none"
                  />
                </svg>
              </button>
            </div>

            {/* Category Items */}
            {!isCollapsed && (
              <div className="flex flex-col space-y-0.5">
                {group.items.map((item) => {
                  const active = isPathActive(currentPath, item.href);
                  return (
                    <a
                      key={item.href}
                      href={item.href}
                      className={`group flex items-center justify-between px-5 py-2 transition-colors border-l-2 ${
                        active
                          ? "text-accent bg-bg-tertiary font-medium border-l-accent"
                          : "text-text-secondary hover:text-text-primary hover:bg-bg-tertiary border-l-transparent"
                      }`}
                    >
                      <span className="truncate">{item.title}</span>
                      {active && (
                        <span className="h-1.5 w-1.5 rounded-full bg-accent shrink-0 ml-2"></span>
                      )}
                    </a>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );
}
