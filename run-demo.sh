#!/usr/bin/env bash
# Quick Demo Runner for AI-Based Visual Detection of Musculoskeletal Disorders

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  AI-Based Visual Detection of Musculoskeletal Disorders with    ║"
echo "║  Personalized Exercise Recommendation                          ║"
echo "║                                                                ║"
echo "║  Quick Demo Runner                                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Check Python environment
echo "Step 1️⃣ : Setting up Python environment..."
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Step 2: Install dependencies
echo "Step 2️⃣ : Installing dependencies..."
.venv/bin/pip install -q numpy matplotlib
echo "   ✓ Dependencies installed"
echo ""

# Step 3: Run the demo
echo "Step 3️⃣ : Running complete dataset & model analysis..."
.venv/bin/python demo.py
echo ""

# Step 4: Open the website
echo "Step 4️⃣ : Opening website in default browser..."
echo ""
echo "   📊 Generated files:"
echo "      • demo_output/dataset_analysis.png"
echo "      • demo_output/model_performance.png"
echo "      • demo_output/body_parts_distribution.png"
echo ""
echo "   🌐 To view the interactive dashboard:"
echo "      • Open: index.html in your browser"
echo "      • Click: 'View Demo Results' to see charts"
echo ""

# Try to open in browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open index.html 2>/dev/null || echo "Open 'index.html' in your web browser"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open index.html 2>/dev/null || echo "Open 'index.html' in your web browser"
else
    # Windows or other
    echo "Open 'index.html' in your web browser"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✓ Demo Complete!                                              ║"
echo "║                                                                ║"
echo "║  You can now show:                                             ║"
echo "║  1. Dataset overview (14,863 studies)                          ║"
echo "║  2. Model performance comparison (13 models)                   ║"
echo "║  3. Body part distribution                                     ║"
echo "║  4. Interactive web dashboard                                  ║"
echo "║  5. PDF report (assets/mura-corrected-report.pdf)              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
