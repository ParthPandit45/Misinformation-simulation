import React, { useCallback, useEffect, useRef } from "react";

/**
 * WorkspaceShell gives the app a research-tool feel:
 * - persistent left navigation
 * - focused content panel on the right
 * - keyboard navigation support
 * - smooth transitions and enhanced accessibility
 */
export default function WorkspaceShell({ tabs, active, onChange, children }) {
  const contentRef = useRef(null);
  const sidebarRef = useRef(null);

  // Scroll content area to top when tab changes
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  }, [active]);

  // Handle keyboard navigation (arrow keys + Enter/Space)
  const handleKeyboardNav = useCallback((e) => {
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
      return;
    }

    e.preventDefault();
    const currentIdx = tabs.findIndex((t) => t.id === active);
    let nextIdx = currentIdx;

    if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      nextIdx = currentIdx > 0 ? currentIdx - 1 : tabs.length - 1;
    } else if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      nextIdx = currentIdx < tabs.length - 1 ? currentIdx + 1 : 0;
    }

    onChange(tabs[nextIdx].id);
  }, [tabs, active, onChange]);

  // Attach keyboard listener to sidebar
  useEffect(() => {
    const ref = sidebarRef.current;
    if (ref) {
      ref.addEventListener("keydown", handleKeyboardNav);
      return () => ref.removeEventListener("keydown", handleKeyboardNav);
    }
  }, [handleKeyboardNav]);

  // Persist active tab to localStorage
  useEffect(() => {
    localStorage.setItem("workspace-active-tab", active);
  }, [active]);

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "230px minmax(0, 1fr)",
      gap: "20px",
      alignItems: "start",
    }}>
      <aside
        ref={sidebarRef}
        role="navigation"
        aria-label="Workspace navigation"
        style={{
          position: "sticky",
          top: "20px",
          background: "var(--color-background-secondary)",
          border: "1px solid var(--color-border-primary)",
          boxShadow: "var(--shadow-premium)",
          borderRadius: "var(--border-radius-lg)",
          padding: "16px",
          outline: "none",
        }}
        tabIndex={0}
      >
        <div style={{ 
          fontSize: "11px", 
          color: "var(--color-text-tertiary)", 
          marginBottom: "12px", 
          letterSpacing: "0.06em", 
          textTransform: "uppercase",
          fontWeight: 700 
        }}>
          Workspace Views
        </div>
        <div style={{ display: "grid", gap: "8px" }}>
          {tabs.map((tab, idx) => {
            const isActive = tab.id === active;
            return (
              <button
                key={tab.id}
                onClick={() => onChange(tab.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onChange(tab.id);
                  }
                }}
                className={`sidebar-tab ${isActive ? "active" : ""}`}
                aria-current={isActive ? "page" : "false"}
                aria-label={tab.description ? `${tab.label}, ${tab.description}` : tab.label}
                title={tab.description || tab.label}
                style={{
                  position: "relative",
                  transition: "all 0.2s ease-out",
                  opacity: isActive ? 1 : 0.7,
                }}
              >
                {tab.icon && (
                  <span style={{ marginRight: "6px", display: "inline-block" }}>
                    {tab.icon}
                  </span>
                )}
                {tab.label}
                {isActive && (
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: "3px",
                      background: "var(--color-accent)",
                      borderRadius: "0 3px 3px 0",
                      animation: "slideIn 0.3s ease-out",
                    }}
                  />
                )}
              </button>
            );
          })}
        </div>
      </aside>

      <section
        ref={contentRef}
        role="main"
        aria-label="Workspace content"
        style={{
          background: "var(--color-background-secondary)",
          border: "1px solid var(--color-border-primary)",
          boxShadow: "var(--shadow-premium)",
          borderRadius: "var(--border-radius-lg)",
          padding: "24px",
          maxHeight: "calc(100vh - 60px)",
          overflowY: "auto",
          animation: "fadeIn 0.2s ease-out",
        }}
      >
        {children}
      </section>

      <style>{`
        @keyframes slideIn {
          from {
            width: 0;
            opacity: 0;
          }
          to {
            width: 3px;
            opacity: 1;
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .sidebar-tab {
          transition: all 0.2s ease-out;
        }

        .sidebar-tab:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .sidebar-tab:focus-visible {
          outline: 2px solid var(--color-accent);
          outline-offset: -2px;
        }
      `}</style>
    </div>
  );
}
