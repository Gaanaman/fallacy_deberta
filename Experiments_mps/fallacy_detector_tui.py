#!/usr/bin/env python3
"""
Technocognitive Fallacy Detector - Terminal UI
===============================================
A beautiful terminal interface for detecting logical fallacies
in climate-related claims using DeBERTa V2 XLarge.

Run with: python fallacy_detector_tui.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, ListView, ListItem
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.console import Console

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = "./best_model"
MAX_SEQ_LENGTH = 512

EXAMPLE_CLAIMS = [
    "It was cold today, so global warming is a hoax.",
    "My uncle is a scientist and he says climate change isn't real.",
    "Climate scientists are just in it for the grant money.",
    "We can't predict the weather next week, so how can we predict climate in 100 years?",
    "The climate has always changed naturally, so humans can't be causing it.",
    "Either we completely stop using fossil fuels or we do nothing about climate change."
]

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print("Loading model... (this may take a moment)")

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
model.zero_grad()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
id2label = model.config.id2label

print(f"Model loaded on {device}!")

# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_and_explain(text: str) -> tuple:
    """
    Predict fallacy type and generate explanations.
    
    Returns:
        Tuple of (prediction_text, highlighted_tokens)
    """
    if not text or text.strip() == "":
        return "Please enter a claim to analyze.", []
    
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
    
    prediction_text = f"Fallacy: {pred_label}\nConfidence: {confidence:.1%}"
    
    return prediction_text, highlighted


# ==============================================================================
# TEXTUAL APP
# ==============================================================================
class FallacyDetectorApp(App):
    """Terminal UI for Fallacy Detection."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
        padding: 1;
    }
    
    #header-text {
        text-align: center;
        padding: 1;
        background: $primary;
        color: $text;
        text-style: bold;
    }
    
    #input-section {
        height: auto;
        max-height: 40%;
        padding: 1;
        border: solid $primary;
    }
    
    #input-area {
        height: 6;
        margin-bottom: 1;
    }
    
    #button-row {
        height: 3;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
    }
    
    #analyze-btn {
        background: $success;
    }
    
    #results-section {
        height: auto;
        min-height: 10;
        padding: 1;
        border: solid $secondary;
        margin-top: 1;
    }
    
    #prediction-label {
        text-style: bold;
        padding: 1;
        background: $primary-darken-2;
    }
    
    #explanation-area {
        height: auto;
        padding: 1;
    }
    
    #examples-section {
        height: auto;
        padding: 1;
        border: solid $accent;
        margin-top: 1;
    }
    
    ListView {
        height: auto;
        max-height: 10;
    }
    
    ListItem {
        padding: 0 1;
    }
    
    ListItem:hover {
        background: $primary-darken-1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("enter", "analyze", "Analyze", show=False),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_claim = ""
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="main-container"):
            yield Static("🕵️ TECHNOCOGNITIVE FALLACY DETECTOR", id="header-text")
            yield Static("DeBERTa-v2-xlarge | Fine-tuned with Focal Loss", 
                        classes="text-center")
            
            # Input Section
            with Vertical(id="input-section"):
                yield Label("📝 Enter a claim to analyze:")
                yield TextArea(id="input-area", soft_wrap=True)
                with Horizontal(id="button-row"):
                    yield Button("🔍 Analyze", id="analyze-btn", variant="success")
                    yield Button("Clear", id="clear-btn", variant="default")
            
            # Results Section
            with Vertical(id="results-section"):
                yield Label("📊 Results", classes="section-title")
                yield Static("Enter a claim and click Analyze", id="prediction-label")
                yield Label("🔬 Explanation (Green=Evidence, Red=Counter):")
                yield Static("", id="explanation-area")
            
            # Examples Section
            with Vertical(id="examples-section"):
                yield Label("💡 Example Claims (click to use):")
                yield ListView(
                    *[ListItem(Label(claim[:60] + "..." if len(claim) > 60 else claim), 
                              id=f"example-{i}") 
                      for i, claim in enumerate(EXAMPLE_CLAIMS)],
                    id="examples-list"
                )
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "analyze-btn":
            self.run_analysis()
        elif event.button.id == "clear-btn":
            self.clear_inputs()
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle example selection."""
        if event.item and event.item.id:
            idx = int(event.item.id.split("-")[1])
            claim = EXAMPLE_CLAIMS[idx]
            self.query_one("#input-area", TextArea).text = claim
    
    def run_analysis(self) -> None:
        """Run the fallacy analysis."""
        input_area = self.query_one("#input-area", TextArea)
        claim = input_area.text
        
        if not claim.strip():
            self.query_one("#prediction-label", Static).update(
                "⚠️ Please enter a claim first!"
            )
            return
        
        # Show loading
        self.query_one("#prediction-label", Static).update("🔄 Analyzing...")
        self.query_one("#explanation-area", Static).update("")
        
        # Run prediction
        try:
            prediction, highlighted = predict_and_explain(claim)
            
            # Update prediction
            self.query_one("#prediction-label", Static).update(prediction)
            
            # Create highlighted text using Rich
            rich_text = Text()
            for token, score in highlighted:
                if score > 0.3:
                    rich_text.append(token, style="bold green")
                elif score > 0.1:
                    rich_text.append(token, style="green")
                elif score < -0.3:
                    rich_text.append(token, style="bold red")
                elif score < -0.1:
                    rich_text.append(token, style="red")
                else:
                    rich_text.append(token)
            
            self.query_one("#explanation-area", Static).update(rich_text)
            
        except Exception as e:
            self.query_one("#prediction-label", Static).update(f"❌ Error: {str(e)}")
    
    def clear_inputs(self) -> None:
        """Clear all inputs and outputs."""
        self.query_one("#input-area", TextArea).text = ""
        self.query_one("#prediction-label", Static).update("Enter a claim and click Analyze")
        self.query_one("#explanation-area", Static).update("")
    
    def action_analyze(self) -> None:
        """Keyboard shortcut for analyze."""
        self.run_analysis()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    app = FallacyDetectorApp()
    app.run()
