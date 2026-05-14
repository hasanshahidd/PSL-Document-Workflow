/* Alpine.js controller for the document workflow UI.
   Talks to the FastAPI surface, builds the live trace Gantt, and
   shows citation source on sentence hover.
*/
function app() {
  return {
    documents: [],
    selected: new Set(),
    draft: null,
    trace: null,
    rules: [],
    stats: { counts: { drafts:0, edits:0, style_rules:0, exemplars:0, traces:0, cost_ledger:0 }, settings: null },
    cost: { calls: 0, input_tokens: 0, output_tokens: 0, usd: 0 },
    demoMode: false,
    busy: { upload: false, draft: false, edit: false },
    editText: "",
    lastEdit: null,
    hoverChunk: null,
    chunkCache: new Map(),
    pickedFile: null,

    async boot() {
      await this.refreshStats();
      await this.refreshRules();
      await this.refreshDocuments();
      this.demoMode = await this.checkDemoMode();
    },

    async refreshDocuments() {
      try {
        const r = await fetch("/documents");
        if (!r.ok) return;
        this.documents = await r.json();
      } catch {}
    },

    async checkDemoMode() {
      try {
        const r = await fetch("/demo-mode");
        if (!r.ok) return false;
        const j = await r.json();
        return Boolean(j.demo_mode);
      } catch { return false; }
    },

    async refreshStats() {
      const r = await fetch("/stats");
      if (!r.ok) return;
      this.stats = await r.json();
      this.cost = this.stats.cost_total || this.cost;
    },

    async refreshRules() {
      const r = await fetch("/style-rules");
      if (r.ok) {
        const j = await r.json();
        this.rules = j.rules || [];
      }
    },

    openFilePicker() {
      // Programmatically click the hidden file input. This MUST run
      // inside the user click handler so the browser counts it as a
      // user gesture and is allowed to show the OS picker.
      const input = this.$refs.file;
      if (!input) {
        alert("File input not found in the page.");
        return;
      }
      input.click();
    },

    onFilePicked() {
      // step 1 of upload. The user has chosen a file but nothing has been
      // sent to the server yet. The Ingest button becomes active.
      const input = this.$refs.file;
      this.pickedFile = input.files && input.files[0] ? input.files[0] : null;
    },

    async ingestPicked() {
      // step 2 of upload. Send the picked file to /ingest.
      const input = this.$refs.file;
      const f = this.pickedFile;
      if (!f) return;
      this.busy.upload = true;
      try {
        const fd = new FormData();
        fd.append("file", f);
        const r = await fetch("/ingest", { method: "POST", body: fd });
        if (!r.ok) throw new Error(await r.text());
        const doc = await r.json();
        // dedup by doc_id (same hash means same file)
        this.documents = [doc, ...this.documents.filter(d => d.doc_id !== doc.doc_id)];
        this.selected.add(doc.doc_id);
        await this.refreshStats();
        this.pickedFile = null;
        input.value = "";
      } catch (e) {
        alert("Ingest failed: " + e.message);
      } finally {
        this.busy.upload = false;
      }
    },

    async deleteDoc(d) {
      if (!confirm(`Remove "${d.filename}"?\n\nThis drops the document's chunks from the vector store and deletes the processed JSON.`)) return;
      try {
        const r = await fetch("/documents/" + encodeURIComponent(d.doc_id), { method: "DELETE" });
        if (!r.ok) throw new Error(await r.text());
        this.documents = this.documents.filter(x => x.doc_id !== d.doc_id);
        this.selected.delete(d.doc_id);
        await this.refreshStats();
      } catch (e) {
        alert("Delete failed: " + e.message);
      }
    },

    toggleSelect(id) {
      if (this.selected.has(id)) this.selected.delete(id);
      else this.selected.add(id);
    },

    async generateDraft() {
      this.busy.draft = true;
      this.trace = null;
      try {
        const r = await fetch("/drafts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doc_ids: [...this.selected] }),
        });
        if (!r.ok) throw new Error(await r.text());
        this.draft = await r.json();
        this.editText = this.draft.sentences.map(s => s.text).join("\n");
        await this.loadTrace(this.draft.draft_id);
        await this.refreshStats();
      } catch (e) {
        alert("Draft failed: " + e.message);
      } finally {
        this.busy.draft = false;
      }
    },

    async loadTrace(draftId) {
      try {
        const r = await fetch("/traces/" + encodeURIComponent(draftId));
        if (!r.ok) { this.trace = null; return; }
        this.trace = await r.json();
      } catch { this.trace = null; }
    },

    totalMs() {
      if (!this.trace || !this.trace.spans || !this.trace.spans.length) return "0";
      const max = Math.max(...this.trace.spans.map(s => s.duration_ms));
      return max.toFixed(1);
    },

    rowStyle(span) {
      if (!this.trace || !this.trace.spans || !this.trace.spans.length) return "";
      const t0 = new Date(this.trace.spans[0].started_at).getTime();
      const ts = new Date(span.started_at).getTime();
      const offset = ts - t0;
      const total = Math.max(...this.trace.spans.map(s =>
        new Date(s.started_at).getTime() - t0 + s.duration_ms
      ));
      const leftPct = total > 0 ? (offset / total) * 100 : 0;
      return `margin-left: ${leftPct.toFixed(2)}%;`;
    },

    barStyle(span) {
      if (!this.trace || !this.trace.spans || !this.trace.spans.length) return "";
      const t0 = new Date(this.trace.spans[0].started_at).getTime();
      const total = Math.max(...this.trace.spans.map(s =>
        new Date(s.started_at).getTime() - t0 + s.duration_ms
      ));
      const widthPct = total > 0 ? (span.duration_ms / total) * 100 : 100;
      return `width: ${Math.max(widthPct, 5).toFixed(2)}%;`;
    },

    sentClass(s) {
      if (s.text.startsWith("#")) return "heading";
      return s.supported ? "" : "unsupported";
    },

    formatSentence(s) {
      // Render [chunk_id] markers as small numbered chips.
      // Each unique chunk_id within a sentence gets its own index.
      const seen = new Map();
      const escape = (str) => str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      return escape(s.text).replace(/\[([A-Za-z0-9_:.\-]+)\]/g, (_, id) => {
        if (!seen.has(id)) seen.set(id, seen.size + 1);
        return ` <span class="cite" title="${escape(id)}">[${seen.get(id)}]</span>`;
      });
    },

    async hoverCite(s) {
      if (!s.citations || s.citations.length === 0) return;
      const c = s.citations[0];
      const cacheKey = c.chunk_id + (c.quote ? '|' + c.quote : '');
      if (this.chunkCache.has(cacheKey)) {
        this.hoverChunk = this.chunkCache.get(cacheKey);
        return;
      }
      let fullText = "(source unavailable)";
      try {
        const r = await fetch("/chunks/" + encodeURIComponent(c.chunk_id));
        if (r.ok) fullText = (await r.json()).text;
      } catch {}
      const chunk = {
        chunk_id: c.chunk_id,
        doc_id: c.doc_id,
        page: c.page,
        text: fullText,
        quote: c.quote || null,
        ocr_confidence: 1.0,
      };
      this.chunkCache.set(cacheKey, chunk);
      this.hoverChunk = chunk;
    },

    async submitEdit() {
      if (!this.draft) return;
      this.busy.edit = true;
      try {
        const r = await fetch(`/drafts/${this.draft.draft_id}/edits`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ edited_text: this.editText }),
        });
        if (!r.ok) throw new Error(await r.text());
        this.lastEdit = await r.json();
        await this.refreshRules();
        await this.refreshStats();
      } catch (e) {
        alert("Edit submission failed: " + e.message);
      } finally {
        this.busy.edit = false;
      }
    },

    confClass(c) {
      if (c >= 0.85) return "high";
      if (c >= 0.6) return "mid";
      return "low";
    },
  };
}
