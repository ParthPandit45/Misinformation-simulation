import React from "react";

/**
 * WorkspaceShell gives the app a research-tool feel:
 * - persistent left navigation
 * - focused content panel on the right
 */
export default function WorkspaceShell({ tabs, active, onChange, children }) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "230px minmax(0, 1fr)",
      gap: "20px",
      alignItems: "start",
    }}>
      <aside style={{
        position: "sticky",
        top: "20px",
        background: "var(--color-background-secondary)",
        border: "1px solid var(--color-border-primary)",
        boxShadow: "var(--shadow-premium)",
        borderRadius: "var(--border-radius-lg)",
        padding: "16px",
      }}>
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
          {tabs.map((tab) => {
            const isActive = tab.id === active;
            return (
              <button
                key={tab.id}
                onClick={() => onChange(tab.id)}
                className={`sidebar-tab ${isActive ? "active" : ""}`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </aside>

      <section style={{
        background: "var(--color-background-secondary)",
        border: "1px solid var(--color-border-primary)",
        boxShadow: "var(--shadow-premium)",
        borderRadius: "var(--border-radius-lg)",
        padding: "24px",
      }}>
        {children}
      </section>
    </div>
  );
}
