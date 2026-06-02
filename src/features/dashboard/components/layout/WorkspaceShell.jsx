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
      gridTemplateColumns: "220px minmax(0, 1fr)",
      gap: "14px",
      alignItems: "start",
    }}>
      <aside style={{
        position: "sticky",
        top: "14px",
        background: "var(--color-background-secondary)",
        border: "1px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "10px",
      }}>
        <div style={{ fontSize: "11px", color: "var(--color-text-tertiary)", marginBottom: "8px", letterSpacing: "0.04em", textTransform: "uppercase" }}>
          Workspace Views
        </div>
        <div style={{ display: "grid", gap: "6px" }}>
          {tabs.map((tab) => {
            const isActive = tab.id === active;
            return (
              <button
                key={tab.id}
                onClick={() => onChange(tab.id)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  borderRadius: "8px",
                  border: isActive ? "1px solid var(--accent)" : "1px solid transparent",
                  background: isActive ? "var(--color-background-primary)" : "transparent",
                  color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  padding: "9px 10px",
                  cursor: "pointer",
                  fontSize: "13px",
                  fontWeight: isActive ? 600 : 500,
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </aside>

      <section style={{
        background: "var(--color-background-secondary)",
        border: "1px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "14px",
      }}>
        {children}
      </section>
    </div>
  );
}
