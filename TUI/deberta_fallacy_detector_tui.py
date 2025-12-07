#!/usr/bin/env python3
"""
Technocognitive Climate Fallacy Detector - Premium Terminal UI
================================================================
A beautiful terminal interface for detecting logical fallacies
in climate-related claims using fine-tuned DeBERTa.

Author: FLICC Research Team
Run with: python deberta_fallacy_detector_tui.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients


from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, ListView, ListItem
from textual.binding import Binding
from rich.text import Text


# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_ID = "Gaanaman/deberta-v2-xlarge-climate-fallacy"
MAX_SEQ_LENGTH = 512

# Fallacy descriptions for user education (12 FLICC classes)
FALLACY_INFO = {
    "ad hominem": {
        "icon": "👤",
        "description": "Attacking the person making the argument rather than the argument itself.",
        "color": "#E74C3C"
    },
    "anecdote": {
        "icon": "📖",
        "description": "Using personal experience or isolated examples instead of sound evidence.",
        "color": "#9B59B6"
    },
    "cherry picking": {
        "icon": "🍒",
        "description": "Selecting only data that supports the argument while ignoring contradicting evidence.",
        "color": "#FF6B6B"
    },
    "conspiracy theory": {
        "icon": "🕵️",
        "description": "Suggesting a secret plot to explain events rather than accepting evidence.",
        "color": "#8E44AD"
    },
    "fake experts": {
        "icon": "🎭",
        "description": "Using an unqualified source or misleading credentials to support a claim.",
        "color": "#4ECDC4"
    },
    "false choice": {
        "icon": "⚖️",
        "description": "Presenting only two options when more alternatives exist.",
        "color": "#3498DB"
    },
    "false equivalence": {
        "icon": "🔄",
        "description": "Treating two things as equal when they are fundamentally different.",
        "color": "#1ABC9C"
    },
    "impossible expectations": {
        "icon": "🎯",
        "description": "Demanding unrealistic precision or certainty from science.",
        "color": "#F39C12"
    },
    "misrepresentation": {
        "icon": "🪞",
        "description": "Distorting or misquoting scientific findings to support a false claim.",
        "color": "#E67E22"
    },
    "oversimplification": {
        "icon": "📉",
        "description": "Reducing a complex issue to a misleadingly simple explanation.",
        "color": "#F1C40F"
    },
    "single cause": {
        "icon": "☝️",
        "description": "Assuming a complex problem has only one cause when multiple factors are involved.",
        "color": "#2ECC71"
    },
    "slothful induction": {
        "icon": "🦥",
        "description": "Ignoring overwhelming evidence and refusing to accept a well-supported conclusion.",
        "color": "#95A5A6"
    }
}

EXAMPLE_CLAIMS = [
    ("Climate scientists can't be trusted because they're just activists.", "ad hominem"),
    ("My grandfather lived through the 1930s dust bowl, so today's climate is nothing new.", "anecdote"),
    ("It was cold today, so global warming is a hoax.", "cherry picking"),
    ("Climate scientists are just in it for the grant money.", "conspiracy theory"),
    ("This retired TV weatherman says climate change isn't real.", "fake experts"),
    ("Either we completely stop using fossil fuels or we do nothing.", "false choice"),
    ("One scientist disagrees, so the science isn't settled.", "false equivalence"),
    ("We can't predict the weather next week, so how can we predict climate?", "impossible expectations"),
    ("Scientists said there would be an ice age in the 70s.", "misrepresentation"),
    ("The climate is just following natural cycles.", "oversimplification"),
    ("Volcanoes emit CO2 too, so they must be causing climate change.", "single cause"),
    ("Despite the evidence, I'm still not convinced humans cause warming.", "slothful induction"),
]


# ==============================================================================
# LOAD MODEL
# ==============================================================================
print("🔄 Loading model... (this may take a moment)")

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
model.zero_grad()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
id2label = model.config.id2label

print(f"✓ Model loaded on {device}!")


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_and_explain(text: str) -> tuple:
    """
    Predict fallacy type and generate explanations.
    
    Returns:
        Tuple of (pred_label, confidence, highlighted_tokens, all_probs)
    """
    if not text or text.strip() == "":
        return None, 0.0, [], {}
    
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    ref_token_id = tokenizer.pad_token_id
    
    # Forward function for Captum
    def model_forward(inputs, mask):
        return model(input_ids=inputs, attention_mask=mask).logits
    
    # Get prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        pred_label = id2label[pred_idx]
        confidence = probabilities[pred_idx].item()
        all_probs = {id2label[i]: probabilities[i].item() for i in range(len(probabilities))}
    
    # Compute attributions
    lig = LayerIntegratedGradients(model_forward, model.deberta.embeddings.word_embeddings)
    
    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=ref_token_id,
        additional_forward_args=(attention_mask,),
        target=pred_idx,
        return_convergence_delta=True
    )
    
    # Process attributions
    attributions = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    attr_max = max(abs(attributions.max()), abs(attributions.min()))
    if attr_max > 0:
        attributions = attributions / attr_max
    
    # Create highlighted tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
    
    highlighted = []
    for token, score in zip(tokens, attributions):
        if token not in special_tokens:
            clean_token = token.replace("▁", " ").replace("Ġ", " ")
            highlighted.append((clean_token, score))
    
    return pred_label, confidence, highlighted, all_probs


# ==============================================================================
# TEXTUAL APP
# ==============================================================================
class ClimateFallacyDetectorApp(App):
    """Retro Terminal UI for Climate Fallacy Detection."""
    
    CSS_PATH = "deberta_fallacy_detector_tui.tcss"
    
    TITLE = "Climate Fallacy Detector"
    SUB_TITLE = "Powered by DeBERTa + Explainable AI"
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+a", "analyze", "Analyze"),
        Binding("ctrl+l", "clear", "Clear"),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_claim = ""
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="app-container"):
            # Header Section
            with Vertical(id="header-section"):
                yield Static(self._render_title(), id="app-title")
                yield Static(f"✓ Model Ready | DeBERTa on {device}", id="model-status")
            
            # Main Content Grid
            with Horizontal(id="content-grid"):
                # Left Panel - Input & Examples
                with Vertical(id="left-panel"):
                    # Input Section
                    with Vertical(id="input-section"):
                        yield Label("Enter a climate claim to analyze:", classes="section-title")
                        yield TextArea(id="input-area", soft_wrap=True)
                        with Horizontal(id="button-row"):
                            yield Button("Analyze", id="analyze-btn", variant="success")
                            yield Button("Clear", id="clear-btn", variant="default")
                    
                    # Examples Section
                    with VerticalScroll(id="examples-section"):
                        yield Label("💡 Example Claims (click to use):", classes="section-title")
                        yield ListView(
                            *[ListItem(
                                Label(f"{FALLACY_INFO.get(fallacy, {'icon': '•'})['icon']} {claim[:50]}..."), 
                                id=f"example-{i}"
                            ) for i, (claim, fallacy) in enumerate(EXAMPLE_CLAIMS)],
                            id="examples-list"
                        )
                
                # Right Panel - Results
                with Vertical(id="right-panel"):
                    # Results Section
                    with VerticalScroll(id="results-section"):
                        yield Label("📊 Analysis Results:", classes="section-title")
                        yield Static("Enter a claim and click Analyze to detect fallacies.", 
                                    id="result-card")
                    
                    # Explanation Section
                    with VerticalScroll(id="explanation-section"):
                        yield Label("Explainability:", classes="section-title")
                        yield Static("Token attributions will appear here after analysis.", 
                                    id="explanation-panel")
        
        yield Footer()
    
    def _render_title(self) -> Text:
        """Render stylish app title."""
        title = Text()
        title.append("🌍 ", style="bold")
        title.append("CLIMATE ", style="bold #4ECDC4")
        title.append("FALLACY ", style="bold #FF6B6B")
        title.append("DETECTOR ", style="bold #9B59B6")
        title.append("🔬", style="bold")
        return title
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "analyze-btn":
            self.run_analysis()
        elif event.button.id == "clear-btn":
            self.action_clear()
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle example selection."""
        if event.item and event.item.id:
            idx = int(event.item.id.split("-")[1])
            claim, _ = EXAMPLE_CLAIMS[idx]
            self.query_one("#input-area", TextArea).text = claim
    
    def action_analyze(self) -> None:
        """Keyboard shortcut for analyze."""
        self.run_analysis()
    
    def run_analysis(self) -> None:
        """Run the fallacy analysis (synchronous)."""
        input_area = self.query_one("#input-area", TextArea)
        claim = input_area.text
        
        if not claim.strip():
            self.query_one("#result-card", Static).update(
                Text("⚠️ Please enter a claim first!", style="yellow italic")
            )
            return
        
        # Show loading
        self.query_one("#result-card", Static).update(
            Text("🔄 Analyzing...", style="cyan italic")
        )
        self.query_one("#explanation-panel", Static).update("")
        
        # Force refresh to show loading state
        self.refresh()
        
        # Run prediction
        try:
            pred_label, confidence, highlighted, all_probs = predict_and_explain(claim)
            
            # Build result display
            result_text = self._build_result_display(pred_label, confidence, all_probs)
            self.query_one("#result-card", Static).update(result_text)
            
            # Build explanation display
            explanation_text = self._build_explanation_display(highlighted)
            self.query_one("#explanation-panel", Static).update(explanation_text)
            
            # Show notification
            info = FALLACY_INFO.get(pred_label, {"icon": "❓"})
            self.notify(
                f"{info['icon']} Detected: {pred_label} ({int(confidence*100)}%)",
                title="Analysis Complete"
            )
            
        except Exception as e:
            self.query_one("#result-card", Static).update(
                Text(f"❌ Error: {str(e)}", style="red")
            )
    
    def _build_result_display(self, pred_label: str, confidence: float, all_probs: dict) -> Text:
        """Build the result card display."""
        info = FALLACY_INFO.get(pred_label, {
            "icon": "❓",
            "description": "Unknown fallacy type.",
            "color": "#888888"
        })
        
        result = Text()
        
        # Main prediction with icon
        result.append(f"\n{info['icon']} ", style="bold")
        result.append(f"{pred_label.upper()}\n", style=f"bold {info['color']}")
        result.append("─" * 40 + "\n", style="dim")
        
        # Confidence meter
        confidence_pct = int(confidence * 100)
        bar_filled = int(confidence * 20)
        bar_empty = 20 - bar_filled
        
        result.append("Confidence: ", style="bold")
        result.append("█" * bar_filled, style=info['color'])
        result.append("░" * bar_empty, style="dim")
        result.append(f" {confidence_pct}%\n\n", style="bold")
        
        # Description
        result.append("📖 ", style="bold")
        result.append(info['description'] + "\n\n", style="italic")
        
        # Top predictions table
        result.append("📊 All Predictions:\n", style="bold")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs[:5]:
            prob_pct = int(prob * 100)
            marker = "▸ " if label == pred_label else "  "
            style = "bold" if label == pred_label else "dim"
            fallacy_icon = FALLACY_INFO.get(label, {"icon": "•"})["icon"]
            result.append(f"{marker}{fallacy_icon} {label}: ", style=style)
            result.append(f"{prob_pct}%\n", style=style)
        
        return result
    
    def _build_explanation_display(self, highlighted: list) -> Text:
        """Build the explanation panel display."""
        if not highlighted:
            return Text("No token attributions available.", style="dim italic")
        
        rich_text = Text()
        rich_text.append("🔬 Token Attribution Analysis\n", style="bold cyan")
        rich_text.append("─" * 40 + "\n\n", style="dim")
        rich_text.append("Legend: ", style="dim")
        rich_text.append("■ Supporting ", style="bold green")
        rich_text.append("■ Opposing ", style="bold red")
        rich_text.append("■ Neutral\n\n", style="dim")
        
        for token, score in highlighted:
            if score > 0.3:
                rich_text.append(token, style="bold green on #1a3a1a")
            elif score > 0.1:
                rich_text.append(token, style="green")
            elif score < -0.3:
                rich_text.append(token, style="bold red on #3a1a1a")
            elif score < -0.1:
                rich_text.append(token, style="red")
            else:
                rich_text.append(token, style="white")
        
        return rich_text
    
    def action_clear(self) -> None:
        """Clear all inputs and outputs."""
        self.query_one("#input-area", TextArea).text = ""
        self.query_one("#result-card", Static).update(
            "Enter a claim and click Analyze to detect fallacies."
        )
        self.query_one("#explanation-panel", Static).update(
            "Token attributions will appear here after analysis."
        )
        self.query_one("#input-area", TextArea).focus()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    app = ClimateFallacyDetectorApp()
    app.run()