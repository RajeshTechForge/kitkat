import { getCollection, render } from "astro:content";

export interface NavItem {
  id: string;
  title: string;
  description?: string;
  href: string;
  category: string;
  order: number;
}

export interface NavGroup {
  category: string;
  items: NavItem[];
}

export interface SearchItem {
  id: string;
  category: string;
  title: string;
  snippet?: string;
  href: string;
  type: "page" | "heading";
}

export interface PaginationLink {
  title: string;
  href: string;
}

export interface DocPagination {
  prev: PaginationLink | null;
  next: PaginationLink | null;
}

export function getDocHref(id: string): string {
  if (id === "index") return "/docs";
  let cleanId = id;
  if (cleanId.endsWith("/index")) {
    cleanId = cleanId.slice(0, -6);
  }
  const segments = cleanId.split("/").map((seg) => encodeURIComponent(seg));
  return `/docs/${segments.join("/")}`;
}

function parseDirectoryCategory(dir: string): string {
  // Strip leading numeric prefixes (e.g., '01-getting-started' -> 'getting-started')
  const cleaned = dir.replace(/^[0-9]+[_-]/, "");
  return cleaned
    .split(/[-_]/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function extractOrderPrefix(segment: string): number | null {
  const match = segment.match(/^([0-9]+)[_-]/);
  return match ? parseInt(match[1], 10) : null;
}

export async function getNavGroups(): Promise<NavGroup[]> {
  const entries = await getCollection("docs");

  const navItems: (NavItem & { categoryOrder: number })[] = entries.map(
    (entry) => {
      const parts = entry.id.split("/");
      const isSubfolder = parts.length > 1;

      let category = entry.data.category;
      let categoryOrder = 999;
      let order = entry.data.order ?? 999;

      if (isSubfolder) {
        const topDir = parts[0];
        const dirOrder = extractOrderPrefix(topDir);
        if (dirOrder !== null) {
          categoryOrder = dirOrder;
        }

        if (!category) {
          category = parseDirectoryCategory(topDir);
        }

        const fileName = parts[parts.length - 1];
        const fileOrder = extractOrderPrefix(fileName);
        if (entry.data.order === undefined && fileOrder !== null) {
          order = fileOrder;
        }
      }

      if (!category) {
        category = "Overview";
        categoryOrder = 0;
      }

      return {
        id: entry.id,
        title: entry.data.title,
        description: entry.data.description,
        href: getDocHref(entry.id),
        category,
        order,
        categoryOrder,
      };
    },
  );

  const groupsMap = new Map<string, { order: number; items: NavItem[] }>();

  for (const item of navItems) {
    if (!groupsMap.has(item.category)) {
      groupsMap.set(item.category, {
        order: item.categoryOrder,
        items: [],
      });
    }
    const group = groupsMap.get(item.category)!;
    group.order = Math.min(group.order, item.categoryOrder);
    group.items.push(item);
  }

  const groups: NavGroup[] = Array.from(groupsMap.entries())
    .map(([category, data]) => {
      data.items.sort(
        (a, b) => a.order - b.order || a.title.localeCompare(b.title),
      );
      return { category, items: data.items, categoryOrder: data.order };
    })
    .sort(
      (a, b) =>
        a.categoryOrder - b.categoryOrder ||
        a.category.localeCompare(b.category),
    );

  return groups.map(({ category, items }) => ({ category, items }));
}

export async function getSortedDocs(): Promise<NavItem[]> {
  const groups = await getNavGroups();
  return groups.flatMap((g) => g.items);
}

export async function getDocPagination(
  currentId: string,
): Promise<DocPagination> {
  const sorted = await getSortedDocs();
  const currentIndex = sorted.findIndex(
    (doc) =>
      doc.id === currentId || getDocHref(doc.id) === getDocHref(currentId),
  );
  if (currentIndex === -1) return { prev: null, next: null };

  const prevDoc = currentIndex > 0 ? sorted[currentIndex - 1] : null;
  const nextDoc =
    currentIndex < sorted.length - 1 ? sorted[currentIndex + 1] : null;

  return {
    prev: prevDoc ? { title: prevDoc.title, href: prevDoc.href } : null,
    next: nextDoc ? { title: nextDoc.title, href: nextDoc.href } : null,
  };
}

export async function getSearchItems(): Promise<SearchItem[]> {
  const sortedDocs = await getSortedDocs();
  const entries = await getCollection("docs");
  const entriesMap = new Map(entries.map((e) => [e.id, e]));

  const searchItems: SearchItem[] = [];

  for (const doc of sortedDocs) {
    const entry = entriesMap.get(doc.id);
    if (!entry) continue;

    searchItems.push({
      id: entry.id,
      category: doc.category,
      title: entry.data.title,
      snippet: entry.data.description,
      href: doc.href,
      type: "page",
    });

    try {
      const { headings } = await render(entry);
      for (const h of headings) {
        if (h.depth === 2 || h.depth === 3) {
          searchItems.push({
            id: `${entry.id}#${h.slug}`,
            category: entry.data.title,
            title: h.text,
            snippet: `Section in ${entry.data.title}`,
            href: `${doc.href}#${h.slug}`,
            type: "heading",
          });
        }
      }
    } catch {
      // Fallback if render fails
    }
  }

  searchItems.push({
    id: "site-homepage",
    category: "Site",
    title: "Homepage",
    snippet: "Go back to main landing page",
    href: "/",
    type: "page",
  });

  return searchItems;
}
