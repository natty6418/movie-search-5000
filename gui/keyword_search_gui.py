#!/usr/bin/env python3
"""Simple Tkinter UI that wraps the keyword search CLI functionality."""

from __future__ import annotations

import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.lib.inverted_index import (
    BM25_B,
    BM25_K1,
    InvertedIndex,
    tokenize,
)


class KeywordSearchGUI:
    """GUI front-end for the keyword search CLI commands."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Keyword Search GUI")
        self.root.geometry("920x700")

        self.inverted_index = InvertedIndex()
        self.index_loaded = False

        self.index_status_var = tk.StringVar(value="Index status: checking cache...")
        self.search_query_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=5)
        self.doc_id_var = tk.StringVar()
        self.term_var = tk.StringVar()
        self.k1_var = tk.DoubleVar(value=BM25_K1)
        self.b_var = tk.DoubleVar(value=BM25_B)

        self._build_layout()
        self.ensure_index_loaded(silent=True)

    def _build_layout(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=20)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)

        index_frame = ttk.LabelFrame(main, text="Inverted Index")
        index_frame.grid(row=0, column=0, sticky="ew")
        index_frame.columnconfigure(1, weight=1)

        build_button = ttk.Button(index_frame, text="Build / Rebuild Index", command=self.build_index)
        build_button.grid(row=0, column=0, padx=(0, 12), pady=10)
        status_label = ttk.Label(index_frame, textvariable=self.index_status_var)
        status_label.grid(row=0, column=1, sticky="w")

        search_frame = ttk.LabelFrame(main, text="Search")
        search_frame.grid(row=1, column=0, pady=15, sticky="ew")
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Query:").grid(row=0, column=0, sticky="w")
        query_entry = ttk.Entry(search_frame, textvariable=self.search_query_var)
        query_entry.grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        query_entry.bind("<Return>", lambda _event: self.run_bm25_search())

        control_frame = ttk.Frame(search_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(control_frame, text="Keyword Search", command=self.run_keyword_search).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(control_frame, text="BM25 Search", command=self.run_bm25_search).grid(row=0, column=1)
        ttk.Label(control_frame, text="Max results:").grid(row=0, column=2, padx=(16, 4))
        limit_spinbox = ttk.Spinbox(control_frame, from_=1, to=50, textvariable=self.limit_var, width=5)
        limit_spinbox.grid(row=0, column=3)

        results_frame = ttk.Frame(main)
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("id", "title", "score"),
            show="headings",
            height=10,
        )
        self.results_tree.heading("id", text="ID")
        self.results_tree.heading("title", text="Title")
        self.results_tree.heading("score", text="Score")
        self.results_tree.column("id", width=60, anchor="center")
        self.results_tree.column("title", width=500, anchor="w")
        self.results_tree.column("score", width=100, anchor="center")
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        stats_frame = ttk.LabelFrame(main, text="Document & Term Stats")
        stats_frame.grid(row=3, column=0, pady=15, sticky="ew")
        for col in range(4):
            stats_frame.columnconfigure(col, weight=1)

        ttk.Label(stats_frame, text="Doc ID:").grid(row=0, column=0, sticky="w")
        ttk.Entry(stats_frame, textvariable=self.doc_id_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(stats_frame, text="Term:").grid(row=0, column=2, sticky="w")
        ttk.Entry(stats_frame, textvariable=self.term_var).grid(row=0, column=3, sticky="ew")

        tf_button = ttk.Button(stats_frame, text="TF", command=self.show_tf)
        tf_button.grid(row=1, column=0, pady=6, sticky="ew")
        tfidf_button = ttk.Button(stats_frame, text="TF-IDF", command=self.show_tfidf)
        tfidf_button.grid(row=1, column=1, pady=6, sticky="ew")
        idf_button = ttk.Button(stats_frame, text="IDF", command=self.show_idf)
        idf_button.grid(row=1, column=2, pady=6, sticky="ew")
        bm25_idf_button = ttk.Button(stats_frame, text="BM25 IDF", command=self.show_bm25_idf)
        bm25_idf_button.grid(row=1, column=3, pady=6, sticky="ew")

        ttk.Label(stats_frame, text="BM25 k1:").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(stats_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.k1_var, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(stats_frame, text="BM25 b:").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(stats_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.b_var, width=6).grid(row=2, column=3, sticky="w")
        ttk.Button(stats_frame, text="BM25 TF", command=self.show_bm25_tf).grid(row=3, column=0, columnspan=4, pady=6, sticky="ew")

        output_frame = ttk.LabelFrame(main, text="Output")
        output_frame.grid(row=4, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(output_frame, height=10, wrap="word", state="disabled")
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scroll.set)

        ttk.Button(output_frame, text="Clear Output", command=lambda: self.write_output("", clear=True)).grid(
            row=1, column=0, columnspan=2, sticky="e", pady=(6, 0)
        )

        main.rowconfigure(2, weight=1)
        main.rowconfigure(4, weight=1)

    def ensure_index_loaded(self, silent: bool = False) -> bool:
        if self.index_loaded:
            return True
        try:
            self.inverted_index.load()
        except FileNotFoundError:
            self.index_loaded = False
            self.index_status_var.set("Index status: missing - build the index first")
            if not silent:
                messagebox.showwarning(
                    "Index missing",
                    "The inverted index cache was not found. Please build the index first.",
                )
            return False
        self.index_loaded = True
        self.index_status_var.set("Index status: loaded from cache")
        return True

    def build_index(self) -> None:
        self.index_status_var.set("Index status: building...")
        self.root.update_idletasks()
        try:
            self.inverted_index = InvertedIndex()
            self.inverted_index.build()
            self.inverted_index.save()
        except Exception as exc:  # noqa: BLE001
            self.index_loaded = False
            self.index_status_var.set("Index status: build failed")
            messagebox.showerror("Build failed", str(exc))
            self.write_output(f"Build failed: {exc}")
            return
        self.index_loaded = True
        self.index_status_var.set("Index status: built and ready")
        messagebox.showinfo("Success", "The inverted index has been rebuilt.")
        self.write_output("Index built successfully.")

    def run_keyword_search(self) -> None:
        query = self.search_query_var.get().strip()
        if not query:
            messagebox.showinfo("Missing query", "Please enter a query to search.")
            return
        if not self.ensure_index_loaded():
            return
        results = []
        for word in query.split():
            doc_ids = self.inverted_index.get_documents(word.lower())
            for doc_id in doc_ids:
                movie = self.inverted_index.docmap.get(doc_id)
                if movie is None:
                    continue
                results.append({"id": doc_id, "title": movie["title"]})
        seen = set()
        unique_results = []
        for item in results:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            unique_results.append(item)
        top_results = unique_results[: self.limit_var.get()]
        self.populate_results([
            {"id": res["id"], "title": res["title"], "score": ""} for res in top_results
        ])
        if not top_results:
            self.write_output(f"No results found for '{query}'.", clear=True)
        else:
            lines = [f"{idx + 1}. {item['id']} - {item['title']}" for idx, item in enumerate(top_results)]
            self.write_output("Keyword search results:\n" + "\n".join(lines), clear=True)

    def run_bm25_search(self) -> None:
        query = self.search_query_var.get().strip()
        if not query:
            messagebox.showinfo("Missing query", "Please enter a query to search.")
            return
        if not self.ensure_index_loaded():
            return
        limit = max(1, self.limit_var.get())
        results = self.inverted_index.bm25_search(query, limit)
        formatted = []
        for doc, score in results:
            formatted.append({
                "id": doc["id"],
                "title": doc["title"],
                "score": f"{score:.2f}",
            })
        self.populate_results(formatted)
        if not formatted:
            self.write_output(f"No BM25 matches found for '{query}'.", clear=True)
        else:
            lines = [
                f"{idx + 1}. {row['title']} (ID {row['id']}) - Score: {row['score']}"
                for idx, row in enumerate(formatted)
            ]
            self.write_output("BM25 results:\n" + "\n".join(lines), clear=True)

    def show_tf(self) -> None:
        payload = self._parse_doc_and_term()
        if payload is None:
            return
        doc_id, term = payload
        if not self.ensure_index_loaded():
            return
        tf = self.inverted_index.get_tf(doc_id, term)
        self.write_output(f"TF of '{term}' in document {doc_id}: {tf}", clear=True)

    def show_idf(self) -> None:
        term = self.term_var.get().strip()
        if not term:
            messagebox.showinfo("Missing term", "Enter a term to calculate IDF.")
            return
        if not self.ensure_index_loaded():
            return
        tokenized_term = tokenize(term)
        if len(tokenized_term) != 1:
            messagebox.showerror("Invalid term", "Please enter a single word for IDF.")
            return
        token = tokenized_term[0]
        total_doc_count = len(self.inverted_index.docmap)
        term_match_doc_count = len(self.inverted_index.get_documents(token))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        self.write_output(
            f"Inverse document frequency of '{term}': {idf:.2f}",
            clear=True,
        )

    def show_tfidf(self) -> None:
        payload = self._parse_doc_and_term()
        if payload is None:
            return
        doc_id, term = payload
        if not self.ensure_index_loaded():
            return
        tokenized_term = tokenize(term)
        if len(tokenized_term) != 1:
            messagebox.showerror("Invalid term", "Please enter a single word for TF-IDF.")
            return
        token = tokenized_term[0]
        tf = self.inverted_index.get_tf(doc_id, token)
        total_doc_count = len(self.inverted_index.docmap)
        term_match_doc_count = len(self.inverted_index.get_documents(token))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        tfidf = tf * idf
        self.write_output(
            f"TF-IDF score of '{term}' in document {doc_id}: {tfidf:.2f}",
            clear=True,
        )

    def show_bm25_idf(self) -> None:
        term = self.term_var.get().strip()
        if not term:
            messagebox.showinfo("Missing term", "Enter a term to calculate BM25 IDF.")
            return
        if not self.ensure_index_loaded():
            return
        bm25_idf = self.inverted_index.get_bm25_idf(term)
        self.write_output(
            f"BM25 IDF score of '{term}': {bm25_idf:.2f}",
            clear=True,
        )

    def show_bm25_tf(self) -> None:
        payload = self._parse_doc_and_term()
        if payload is None:
            return
        doc_id, term = payload
        if not self.ensure_index_loaded():
            return
        try:
            k1 = float(self.k1_var.get())
            b_value = float(self.b_var.get())
        except tk.TclError:
            messagebox.showerror("Invalid parameters", "k1 and b must be numbers.")
            return
        bm25_tf = self.inverted_index.get_bm25_tf(doc_id, term, k1=k1, b=b_value)
        self.write_output(
            f"BM25 TF score of '{term}' in document {doc_id}: {bm25_tf:.2f}",
            clear=True,
        )

    def _parse_doc_and_term(self) -> tuple[int, str] | None:
        doc_id_value = self.doc_id_var.get().strip()
        term = self.term_var.get().strip()
        if not doc_id_value:
            messagebox.showinfo("Missing document ID", "Enter a document ID.")
            return None
        if not term:
            messagebox.showinfo("Missing term", "Enter a term.")
            return None
        try:
            doc_id = int(doc_id_value)
        except ValueError:
            messagebox.showerror("Invalid document ID", "Document ID must be an integer.")
            return None
        return doc_id, term

    def populate_results(self, rows: list[dict[str, str]]) -> None:
        for child in self.results_tree.get_children():
            self.results_tree.delete(child)
        for row in rows:
            self.results_tree.insert("", "end", values=(row.get("id"), row.get("title"), row.get("score", "")))

    def write_output(self, message: str, clear: bool = False) -> None:
        self.output_text.configure(state="normal")
        if clear:
            self.output_text.delete("1.0", tk.END)
        if message:
            self.output_text.insert(tk.END, message + ("\n" if not message.endswith("\n") else ""))
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    KeywordSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
