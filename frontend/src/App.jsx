import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const FONT_UI =
  "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
const FONT_MONO =
  "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

const COLORS = {
  ink: "#0F172A",
  inkSoft: "#5B6B7F",
  bg: "#F5F7FA",
  surface: "#FFFFFF",
  border: "#E2E8F0",
  primary: "#0B5FA5",
  primarySoft: "#EAF2FA",
  teal: "#0E7C7B",
  alert: "#B42318",
  alertSoft: "#FDEDEC",
};

export default function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! I am your Clinical AI Specialist. Attach a report (PDF or image) and ask a question about it, or just ask a general health question.",
    },
  ]);
  const [input, setInput] = useState("");
  const [attachedFile, setAttachedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  const handleFilePick = (e) => {
    const file = e.target.files[0];
    if (file) setAttachedFile(file);
    e.target.value = "";
  };

  const clearAttachment = () => setAttachedFile(null);

  const sendMessage = async () => {
    const text = input.trim();
    const file = attachedFile;
    if (!text && !file) return;

    const userLabel = file
      ? `📎 ${file.name}${text ? `\n\n${text}` : ""}`
      : text;

    setMessages((prev) => [...prev, { sender: "user", text: userLabel }]);
    setInput("");
    setAttachedFile(null);
    setLoading(true);

    try {
      let response;
      if (file) {
        // Unified Upload & Query: file + question travel together in one
        // multipart/form-data request to /upload.
        const formData = new FormData();
        formData.append("file", file);
        if (text) formData.append("msg", text);
        response = await fetch("http://localhost:8080/upload", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("http://localhost:8080/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ msg: text }),
        });
      }

      const data = await response.json();
      const botResponse =
        data.response || data.message || "I've processed your request.";
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: botResponse,
          warning: !!data.guardrail_triggered,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Connection error. Check your Flask backend!",
          warning: true,
        },
      ]);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={styles.page}>
      <style>{KEYFRAMES}</style>

      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logoMark}>+</div>
          <div>
            <div style={styles.title}>Clinical AI Assistant</div>
            <div style={styles.subtitle}>
              Report analysis &amp; clinical Q&amp;A
            </div>
          </div>
        </div>
        <div style={styles.statusPill}>
          <span style={styles.statusDot} />
          Model ready
        </div>
      </header>

      <div style={styles.chatWindow} ref={scrollRef}>
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {loading && (
          <div
            style={{
              ...styles.bubble,
              ...styles.botBubble,
              animation: "riseIn 200ms ease-out",
            }}
          >
            <ThinkingDots />
          </div>
        )}
      </div>

      <div style={styles.inputArea}>
        {attachedFile && (
          <div style={styles.attachChip}>
            <span style={styles.attachIcon}>📎</span>
            <span style={styles.attachName}>{attachedFile.name}</span>
            <button
              style={styles.attachRemove}
              onClick={clearAttachment}
              aria-label="Remove attachment"
            >
              ×
            </button>
          </div>
        )}
        <div style={styles.inputRow}>
          <input
            type="file"
            ref={fileInputRef}
            accept=".pdf,image/*"
            style={{ display: "none" }}
            onChange={handleFilePick}
          />
          <button
            style={styles.uploadBtn}
            onClick={() => fileInputRef.current.click()}
            title="Attach a report (PDF or image)"
            aria-label="Attach file"
          >
            📁
          </button>
          <textarea
            rows={1}
            style={styles.input}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              attachedFile
                ? "Ask something about this report… (optional)"
                : "Ask a clinical question or attach a report…"
            }
          />
          <button
            style={{
              ...styles.sendBtn,
              ...(loading || (!input.trim() && !attachedFile)
                ? styles.sendBtnDisabled
                : {}),
            }}
            onClick={sendMessage}
            disabled={loading || (!input.trim() && !attachedFile)}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div style={styles.thinkingRow}>
      <span style={{ ...styles.dot, animationDelay: "0ms" }} />
      <span style={{ ...styles.dot, animationDelay: "120ms" }} />
      <span style={{ ...styles.dot, animationDelay: "240ms" }} />
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.sender === "user";
  const isWarning = !!message.warning;

  return (
    <div
      style={{
        ...styles.bubbleRow,
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      {!isUser && <div style={styles.avatarBot}>AI</div>}
      <div
        style={{
          ...styles.bubble,
          ...(isUser ? styles.userBubble : styles.botBubble),
          ...(isWarning ? styles.warningBubble : {}),
        }}
      >
        <ReactMarkdown
          components={{
            table: ({ node, ...props }) => (
              <div style={styles.tableWrap}>
                <table style={styles.table} {...props} />
              </div>
            ),
            thead: ({ node, ...props }) => (
              <thead style={styles.thead} {...props} />
            ),
            th: ({ node, ...props }) => <th style={styles.th} {...props} />,
            td: ({ node, ...props }) => <td style={styles.td} {...props} />,
            tr: ({ node, ...props }) => <tr style={styles.tr} {...props} />,
            h2: ({ node, ...props }) => (
              <h2 style={styles.sectionHeading} {...props} />
            ),
            h3: ({ node, ...props }) => (
              <h3 style={styles.sectionHeading} {...props} />
            ),
            strong: ({ node, ...props }) => (
              <strong style={{ color: COLORS.teal }} {...props} />
            ),
          }}
        >
          {message.text}
        </ReactMarkdown>
      </div>
      {isUser && <div style={styles.avatarUser}>You</div>}
    </div>
  );
}

const KEYFRAMES = `
@keyframes riseIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseDot {
  0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
  40% { opacity: 1; transform: scale(1); }
}
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.001ms !important; }
}
`;

const styles = {
  page: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: COLORS.bg,
    fontFamily: FONT_UI,
    color: COLORS.ink,
  },
  header: {
    padding: "14px 22px",
    background: COLORS.surface,
    borderBottom: `1px solid ${COLORS.border}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerLeft: { display: "flex", alignItems: "center", gap: 12 },
  logoMark: {
    width: 34,
    height: 34,
    borderRadius: 9,
    background: COLORS.primary,
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: 18,
    flexShrink: 0,
  },
  title: { fontWeight: 700, fontSize: 15.5, letterSpacing: "-0.01em" },
  subtitle: { fontSize: 12, color: COLORS.inkSoft, marginTop: 1 },
  statusPill: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 12,
    color: COLORS.inkSoft,
    background: COLORS.bg,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 999,
    padding: "5px 10px",
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#16A34A",
    display: "inline-block",
  },
  chatWindow: {
    flex: 1,
    padding: "22px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 14,
    overflowY: "auto",
  },
  bubbleRow: {
    display: "flex",
    alignItems: "flex-end",
    gap: 8,
    animation: "riseIn 220ms ease-out",
  },
  avatarBot: {
    width: 26,
    height: 26,
    borderRadius: "50%",
    background: COLORS.teal,
    color: "#fff",
    fontSize: 10.5,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  avatarUser: {
    width: 26,
    height: 26,
    borderRadius: "50%",
    background: COLORS.border,
    color: COLORS.inkSoft,
    fontSize: 9.5,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  bubble: {
    padding: "13px 16px",
    borderRadius: 14,
    maxWidth: "72%",
    fontSize: 14.5,
    lineHeight: 1.65,
    whiteSpace: "pre-wrap",
  },
  userBubble: {
    background: COLORS.primary,
    color: "#fff",
    borderBottomRightRadius: 4,
  },
  botBubble: {
    background: COLORS.surface,
    color: COLORS.ink,
    border: `1px solid ${COLORS.border}`,
    borderLeft: `3px solid ${COLORS.teal}`,
    borderBottomLeftRadius: 4,
    boxShadow: "0 1px 2px rgba(15, 23, 42, 0.04)",
  },
  warningBubble: {
    borderLeft: `3px solid ${COLORS.alert}`,
    background: COLORS.alertSoft,
  },
  thinkingRow: { display: "flex", gap: 5, padding: "2px 2px" },
  dot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: COLORS.teal,
    display: "inline-block",
    animation: "pulseDot 1.1s ease-in-out infinite",
  },
  tableWrap: {
    overflowX: "auto",
    marginTop: 10,
    marginBottom: 4,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
  },
  table: { borderCollapse: "collapse", width: "100%", fontSize: 13 },
  thead: { background: COLORS.primarySoft },
  th: {
    textAlign: "left",
    padding: "8px 12px",
    color: COLORS.teal,
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    borderBottom: `1px solid ${COLORS.border}`,
  },
  tr: { borderBottom: `1px solid ${COLORS.border}` },
  td: {
    padding: "8px 12px",
    fontFamily: FONT_MONO,
    fontSize: 12.5,
    color: COLORS.ink,
  },
  sectionHeading: {
    fontSize: 13,
    fontWeight: 700,
    color: COLORS.teal,
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    margin: "14px 0 6px",
  },
  inputArea: {
    padding: "14px 20px 18px",
    background: COLORS.surface,
    borderTop: `1px solid ${COLORS.border}`,
  },
  attachChip: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    background: COLORS.primarySoft,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 999,
    padding: "5px 8px 5px 12px",
    fontSize: 12.5,
    color: COLORS.primary,
    marginBottom: 8,
    animation: "riseIn 150ms ease-out",
  },
  attachIcon: { fontSize: 13 },
  attachName: {
    maxWidth: 220,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  attachRemove: {
    border: "none",
    background: "transparent",
    color: COLORS.inkSoft,
    fontSize: 16,
    lineHeight: 1,
    cursor: "pointer",
    padding: "0 2px",
  },
  inputRow: { display: "flex", gap: 10, alignItems: "flex-end" },
  input: {
    flex: 1,
    padding: "12px 16px",
    borderRadius: 10,
    border: `1px solid ${COLORS.border}`,
    outline: "none",
    fontFamily: FONT_UI,
    fontSize: 14.5,
    resize: "none",
    maxHeight: 120,
  },
  sendBtn: {
    padding: "11px 20px",
    background: COLORS.primary,
    color: "white",
    border: "none",
    borderRadius: 10,
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 14,
  },
  sendBtnDisabled: {
    background: COLORS.border,
    color: COLORS.inkSoft,
    cursor: "not-allowed",
  },
  uploadBtn: {
    padding: "10px 12px",
    background: COLORS.bg,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 10,
    cursor: "pointer",
    fontSize: 15,
  },
};
