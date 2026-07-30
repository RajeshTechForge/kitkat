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
  return `/docs/${encodeURIComponent(id)}`;
}

export async function getNavGroups(): Promise<NavGroup[]> {
  const entries = await getCollection("docs");

  const navItems: NavItem[] = entries.map((entry) => ({
    id: entry.id,
    title: entry.data.title,
    description: entry.data.description,
    href: getDocHref(entry.id),
    category: entry.data.category || "Overview",
    order: entry.data.order ?? 0,
  }));

  const groupsMap = new Map<string, NavItem[]>();
  for (const item of navItems) {
    if (!groupsMap.has(item.category)) {
      groupsMap.set(item.category, []);
    }
    groupsMap.get(item.category)!.push(item);
  }

  const groups: NavGroup[] = Array.from(groupsMap.entries()).map(
    ([category, items]) => {
      items.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
      return { category, items };
    },
  );

  groups.sort((a, b) => {
    const minA = Math.min(...a.items.map((i) => i.order));
    const minB = Math.min(...b.items.map((i) => i.order));
    return minA - minB;
  });

  return groups;
}

export async function getSortedDocs(): Promise<NavItem[]> {
  const groups = await getNavGroups();
  return groups.flatMap((g) => g.items);
}

export async function getDocPagination(
  currentId: string,
): Promise<DocPagination> {
  const sorted = await getSortedDocs();
  const currentIndex = sorted.findIndex((doc) => doc.id === currentId);
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
  const entries = await getCollection("docs");
  const searchItems: SearchItem[] = [];

  for (const entry of entries) {
    const href = getDocHref(entry.id);
    const category = entry.data.category || "Docs";

    searchItems.push({
      id: entry.id,
      category,
      title: entry.data.title,
      snippet: entry.data.description,
      href,
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
            href: `${href}#${h.slug}`,
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
