import { useState } from "react";

interface Node {
  label: string;
  href: string;
  children?: Node[];
}

const TREE: Node[] = [
  { label: "Introduction", href: "/docs" },
  { label: "Installation", href: "/docs/installation" },
  { label: "Core Concepts", href: "/docs/core-concepts" },
  {
    label: "API Reference",
    href: "/docs/api-reference",
    children: [
      { label: "pipeline()", href: "/docs/api-reference#pipeline" },
      { label: "stage()", href: "/docs/api-reference#stage" },
      { label: "exceptions", href: "/docs/api-reference#exceptions" },
    ],
  },
  { label: "Examples & Tutorials", href: "/docs/examples" },
  { label: "Contributing", href: "/docs/contributing" },
  { label: "Changelog", href: "/docs/changelog" },
];

function isActive(currentPath: string, href: string) {
  const clean = href.split("#")[0];
  if (clean === "/docs") return currentPath === "/docs";
  return currentPath === clean;
}

function containsActive(node: Node, currentPath: string): boolean {
  if (isActive(currentPath, node.href)) return true;
  return !!node.children?.some((c) => isActive(currentPath, c.href));
}

export default function DocsSidebar({ currentPath }: { currentPath: string }) {
  const [openMap, setOpenMap] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    for (const node of TREE) {
      if (node.children)
        initial[node.label] = containsActive(node, currentPath);
    }
    return initial;
  });

  return (
    <nav className="flex flex-col text-[13.5px]">
      {TREE.map((node) => {
        const active = isActive(currentPath, node.href);
        const hasChildren = !!node.children?.length;
        const open = openMap[node.label];

        return (
          <div
            key={node.label}
            className="border-b border-border last:border-b-0"
          >
            <div className="flex items-stretch">
              <a
                href={node.href}
                className={`flex-1 px-5 py-3 transition-colors ${
                  active
                    ? "text-accent bg-bg-tertiary border-l-2 border-l-accent"
                    : "text-text-secondary hover:text-text-primary hover:bg-bg-tertiary border-l-2 border-l-transparent"
                }`}
              >
                {node.label}
              </a>
              {hasChildren && (
                <button
                  aria-label={open ? "Collapse section" : "Expand section"}
                  onClick={() =>
                    setOpenMap((m) => ({ ...m, [node.label]: !m[node.label] }))
                  }
                  className="px-4 text-text-tertiary hover:text-text-primary transition-colors"
                >
                  <svg
                    width="9"
                    height="9"
                    viewBox="0 0 9 9"
                    className={`transition-transform ${open ? "rotate-90" : ""}`}
                  >
                    <path
                      d="M2 1L6.5 4.5L2 8"
                      stroke="currentColor"
                      strokeWidth="1.3"
                      fill="none"
                    />
                  </svg>
                </button>
              )}
            </div>
            {hasChildren && open && (
              <div className="flex flex-col bg-bg-primary">
                {node.children!.map((child) => {
                  const childActive = isActive(currentPath, child.href);
                  return (
                    <a
                      key={child.href}
                      href={child.href}
                      className={`font-mono text-[12.5px] pl-9 pr-5 py-2.5 border-t border-border transition-colors ${
                        childActive
                          ? "text-accent bg-bg-tertiary"
                          : "text-text-tertiary hover:text-text-primary hover:bg-bg-tertiary"
                      }`}
                    >
                      {child.label}
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
