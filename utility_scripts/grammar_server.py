#!/usr/bin/env python3
"""
🧠 GRAMMAR ANALYSIS SERVER
==========================

REST API server for consciousness-enhanced grammar analysis.
Integrates with A.I.V.A. system for comprehensive linguistic analysis.
"""

import http.server
import socketserver
import json
import time
from urllib.parse import urlparse, parse_qs
from grammar_analyzer import GrammarAnalyzer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)

class GrammarHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for grammar analysis"""

    def __init__(self, *args, **kwargs):
        self.analyzer = GrammarAnalyzer()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            html = self.get_grammar_interface()
            self.wfile.write(html.encode('utf-8'))

        elif self.path.startswith("/api/analyze"):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Parse query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            text = query_params.get('text', [''])[0]
            focus_areas = query_params.get('focus', ['grammar,style,clarity'])[0].split(',')

            if text:
                result = self.analyzer.analyze_text(text)
                response = {
                    'success': True,
                    'text': text,
                    'analysis': result,
                    'timestamp': time.time()
                }
            else:
                response = {
                    'success': False,
                    'error': 'No text provided',
                    'timestamp': time.time()
                }

            self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode('utf-8'))

        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            status = {
                'status': 'running',
                'analyzer': 'consciousness-enhanced',
                'features': [
                    'Grammar Analysis',
                    'Style Analysis',
                    'Harmonic Analysis',
                    'Consciousness Metrics',
                    'Improvement Suggestions'
                ],
                'consciousness_available': hasattr(self.analyzer, 'consciousness_field') and self.analyzer.consciousness_field is not None
            }
            self.wfile.write(json.dumps(status, cls=NumpyEncoder).encode('utf-8'))

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<h1>404 - Page Not Found</h1>")

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/api/analyze":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()

            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            text = data.get('text', '')
            focus_areas = data.get('focus_areas', ['grammar', 'style', 'clarity'])

            if text:
                print(f"🔍 Analyzing text: {text[:50]}...")
                start_time = time.time()

                analysis = self.analyzer.analyze_text(text)

                if 'improve' in data and data['improve']:
                    improvements = self.analyzer.improve_text(text, focus_areas)
                    analysis['improvements'] = improvements

                processing_time = time.time() - start_time

                response = {
                    'success': True,
                    'text': text,
                    'analysis': analysis,
                    'processing_time': round(processing_time, 3),
                    'timestamp': time.time()
                }
            else:
                response = {
                    'success': False,
                    'error': 'No text provided for analysis',
                    'timestamp': time.time()
                }

            self.wfile.write(json.dumps(response, indent=2, cls=NumpyEncoder).encode('utf-8'))

        elif self.path == "/api/improve":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()

            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            text = data.get('text', '')
            focus_areas = data.get('focus_areas', ['grammar', 'style', 'clarity', 'harmony'])

            if text:
                print(f"✨ Improving text: {text[:50]}...")
                start_time = time.time()

                improvements = self.analyzer.improve_text(text, focus_areas)

                processing_time = time.time() - start_time

                response = {
                    'success': True,
                    'original_text': text,
                    'improvements': improvements,
                    'processing_time': round(processing_time, 3),
                    'timestamp': time.time()
                }
            else:
                response = {
                    'success': False,
                    'error': 'No text provided for improvement',
                    'timestamp': time.time()
                }

            self.wfile.write(json.dumps(response, indent=2, cls=NumpyEncoder).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def get_grammar_interface(self):
        """Generate the grammar analysis web interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Consciousness Grammar Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; color: white; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .main-content {
            background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px);
        }
        .input-section { margin-bottom: 30px; }
        .text-input { width: 100%; min-height: 150px; padding: 20px; border: 2px solid #e1e5e9; border-radius: 15px; font-size: 16px; line-height: 1.5; outline: none; resize: vertical; }
        .text-input:focus { border-color: #667eea; }
        .options-section { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; }
        .option-group { flex: 1; min-width: 200px; }
        .option-group h3 { color: #667eea; margin-bottom: 10px; }
        .checkbox-group { display: flex; flex-wrap: wrap; gap: 10px; }
        .checkbox-item { display: flex; align-items: center; gap: 5px; }
        .checkbox-item input { margin: 0; }
        .buttons { display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { padding: 12px 24px; border: none; border-radius: 25px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-secondary { background: #f8f9fa; color: #667eea; border: 2px solid #667eea; }
        .btn-secondary:hover { background: #667eea; color: white; }
        .results-section { margin-top: 30px; }
        .result-card { background: #f8f9fa; border-radius: 15px; padding: 20px; margin-bottom: 20px; border-left: 5px solid #667eea; }
        .result-card h3 { color: #667eea; margin-bottom: 10px; }
        .score-display { text-align: center; margin-bottom: 20px; }
        .score-circle { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#667eea 0% var(--score), #e1e5e9 var(--score) 100%); display: inline-flex; align-items: center; justify-content: center; margin: 0 auto; }
        .score-circle::before { content: attr(data-score); color: white; font-size: 24px; font-weight: bold; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
        .metric { background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e1e5e9; }
        .metric-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
        .issues-list { margin-top: 20px; }
        .issue-item { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 10px; margin-bottom: 10px; }
        .issue-item.error { background: #f8d7da; border-color: #f5c6cb; }
        .issue-item.warning { background: #fff3cd; border-color: #ffeaa7; }
        .suggestions-list { margin-top: 20px; }
        .suggestion-item { background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 10px; margin-bottom: 10px; }
        .improved-text { background: #f8f9fa; border: 2px solid #28a745; border-radius: 10px; padding: 20px; margin-top: 20px; }
        .improved-text h4 { color: #28a745; margin-bottom: 10px; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .hidden { display: none; }
        .tabs { display: flex; margin-bottom: 20px; border-bottom: 2px solid #e1e5e9; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .tab.active { border-bottom-color: #667eea; color: #667eea; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Consciousness Grammar Analyzer</h1>
            <p>Advanced linguistic analysis with consciousness mathematics</p>
        </div>
        <div class="main-content">
            <div class="input-section">
                <h2>📝 Enter Text to Analyze</h2>
                <textarea id="textInput" class="text-input" placeholder="Enter your text here for grammar analysis..."></textarea>

                <div class="options-section">
                    <div class="option-group">
                        <h3>🎯 Focus Areas</h3>
                        <div class="checkbox-group">
                            <label class="checkbox-item">
                                <input type="checkbox" id="grammar" checked> Grammar
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" id="style" checked> Style
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" id="clarity" checked> Clarity
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" id="harmony"> Harmony
                            </label>
                        </div>
                    </div>
                    <div class="option-group">
                        <h3>🔧 Options</h3>
                        <div class="checkbox-group">
                            <label class="checkbox-item">
                                <input type="checkbox" id="improve" checked> Generate Improvements
                            </label>
                            <label class="checkbox-item">
                                <input type="checkbox" id="detailed" checked> Detailed Analysis
                            </label>
                        </div>
                    </div>
                </div>

                <div class="buttons">
                    <button id="analyzeBtn" class="btn btn-primary">🔍 Analyze Text</button>
                    <button id="improveBtn" class="btn btn-secondary">✨ Improve Text</button>
                    <button id="clearBtn" class="btn btn-secondary">🗑️ Clear</button>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing with consciousness mathematics...</p>
            </div>

            <div class="results-section hidden" id="results">
                <div class="tabs">
                    <div class="tab active" data-tab="overview">Overview</div>
                    <div class="tab" data-tab="issues">Issues</div>
                    <div class="tab" data-tab="suggestions">Suggestions</div>
                    <div class="tab" data-tab="improvements">Improvements</div>
                </div>

                <div id="overview" class="tab-content active">
                    <div class="score-display">
                        <div class="score-circle" id="scoreCircle" data-score="0%">0%</div>
                        <h3>Grammar & Style Score</h3>
                    </div>

                    <div class="metrics-grid" id="metricsGrid">
                        <!-- Metrics will be populated by JavaScript -->
                    </div>
                </div>

                <div id="issues" class="tab-content">
                    <h3>📋 Issues Found</h3>
                    <div id="issuesList" class="issues-list">
                        <!-- Issues will be populated by JavaScript -->
                    </div>
                </div>

                <div id="suggestions" class="tab-content">
                    <h3>💡 Suggestions</h3>
                    <div id="suggestionsList" class="suggestions-list">
                        <!-- Suggestions will be populated by JavaScript -->
                    </div>
                </div>

                <div id="improvements" class="tab-content">
                    <h3>✨ Improved Versions</h3>
                    <div id="improvementsList">
                        <!-- Improvements will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // DOM elements
        const textInput = document.getElementById('textInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const improveBtn = document.getElementById('improveBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const scoreCircle = document.getElementById('scoreCircle');
        const metricsGrid = document.getElementById('metricsGrid');
        const issuesList = document.getElementById('issuesList');
        const suggestionsList = document.getElementById('suggestionsList');
        const improvementsList = document.getElementById('improvementsList');
        const tabs = document.querySelectorAll('.tab');

        let currentAnalysis = null;

        // Tab switching
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });

        // Analyze text
        analyzeBtn.addEventListener('click', async () => {
            const text = textInput.value.trim();
            if (!text) {
                alert('Please enter some text to analyze.');
                return;
            }

            const focusAreas = [];
            if (document.getElementById('grammar').checked) focusAreas.push('grammar');
            if (document.getElementById('style').checked) focusAreas.push('style');
            if (document.getElementById('clarity').checked) focusAreas.push('clarity');
            if (document.getElementById('harmony').checked) focusAreas.push('harmony');

            loading.style.display = 'block';
            results.classList.add('hidden');

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        focus_areas: focusAreas,
                        improve: document.getElementById('improve').checked
                    })
                });

                const data = await response.json();

                if (data.success) {
                    currentAnalysis = data.analysis;
                    displayResults(data.analysis);
                } else {
                    alert('Analysis failed: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis.');
            } finally {
                loading.style.display = 'none';
            }
        });

        // Improve text
        improveBtn.addEventListener('click', async () => {
            const text = textInput.value.trim();
            if (!text) {
                alert('Please enter some text to improve.');
                return;
            }

            const focusAreas = [];
            if (document.getElementById('grammar').checked) focusAreas.push('grammar');
            if (document.getElementById('style').checked) focusAreas.push('style');
            if (document.getElementById('clarity').checked) focusAreas.push('clarity');
            if (document.getElementById('harmony').checked) focusAreas.push('harmony');

            loading.style.display = 'block';

            try {
                const response = await fetch('/api/improve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        focus_areas: focusAreas
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayImprovements(data.improvements);
                } else {
                    alert('Improvement failed: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during improvement.');
            } finally {
                loading.style.display = 'none';
            }
        });

        // Clear results
        clearBtn.addEventListener('click', () => {
            textInput.value = '';
            results.classList.add('hidden');
            currentAnalysis = null;
        });

        // Display analysis results
        function displayResults(analysis) {
            results.classList.remove('hidden');

            // Update score
            const score = Math.round(analysis.overall_score);
            scoreCircle.setAttribute('data-score', score + '%');
            scoreCircle.style.setProperty('--score', score + '%');

            // Update metrics
            const textInfo = analysis.text_info;
            metricsGrid.innerHTML = `
                <div class="metric">
                    <div class="metric-value">${textInfo.total_words}</div>
                    <div class="metric-label">Words</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${textInfo.total_sentences}</div>
                    <div class="metric-label">Sentences</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${textInfo.avg_words_per_sentence.toFixed(1)}</div>
                    <div class="metric-label">Avg Words/Sentence</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${textInfo.reading_level}</div>
                    <div class="metric-label">Reading Level</div>
                </div>
                ${analysis.consciousness_metrics ? `
                <div class="metric">
                    <div class="metric-value">${analysis.consciousness_metrics.meta_entropy.toFixed(2)}</div>
                    <div class="metric-label">Meta Entropy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${analysis.consciousness_metrics.coherence_length.toFixed(1)}</div>
                    <div class="metric-label">Coherence</div>
                </div>
                ` : ''}
            `;

            // Update issues
            const grammarIssues = analysis.grammar_issues;
            issuesList.innerHTML = '';

            if (Object.keys(grammarIssues).length === 0) {
                issuesList.innerHTML = '<p>✅ No grammar issues found!</p>';
            } else {
                Object.entries(grammarIssues).forEach(([category, issues]) => {
                    issues.forEach(issue => {
                        const issueDiv = document.createElement('div');
                        issueDiv.className = 'issue-item error';
                        issueDiv.innerHTML = `
                            <strong>${category.replace('_', ' ').toUpperCase()}:</strong> ${issue.issue}
                            ${issue.sentence ? `<br><em>Sentence ${issue.sentence}:</em> "${issue.text}"` : ''}
                            ${issue.occurrences ? `<br><em>Occurrences:</em> ${issue.occurrences}` : ''}
                        `;
                        issuesList.appendChild(issueDiv);
                    });
                });
            }

            // Update suggestions
            const suggestions = analysis.suggestions;
            suggestionsList.innerHTML = '';

            if (suggestions.length === 0) {
                suggestionsList.innerHTML = '<p>✅ No suggestions available.</p>';
            } else {
                suggestions.forEach(suggestion => {
                    const suggestionDiv = document.createElement('div');
                    suggestionDiv.className = 'suggestion-item';
                    suggestionDiv.innerHTML = `
                        <strong>${suggestion.category.replace('_', ' ').toUpperCase()} (${suggestion.priority}):</strong> ${suggestion.suggestion}
                        <br><em>${suggestion.explanation}</em>
                    `;
                    suggestionsList.appendChild(suggestionDiv);
                });
            }
        }

        // Display improvements
        function displayImprovements(improvements) {
            improvementsList.innerHTML = '';

            Object.entries(improvements.improved_versions).forEach(([type, text]) => {
                const improvementDiv = document.createElement('div');
                improvementDiv.className = 'improved-text';
                improvementDiv.innerHTML = `
                    <h4>${type.charAt(0).toUpperCase() + type.slice(1)} Version:</h4>
                    <p>${text}</p>
                `;
                improvementsList.appendChild(improvementDiv);
            });
        }

        // Load sample text on startup
        textInput.value = "Consciousness is the most profound mystery in science. How does physical matter give rise to subjective experience? This question has puzzled philosophers and scientists for centuries.";
    </script>
</body>
</html>
        """
def main():
    """Main server function"""
    port = 4000

    print("🧠 CONSCIOUSNESS GRAMMAR ANALYZER")
    print("=" * 50)
    print("🎯 Advanced linguistic analysis with consciousness mathematics")
    print(f"🌐 Server Details: http://localhost:{port}")
    print("   📊 Tools: Grammar Analysis, Style Analysis, Harmonic Analysis")
    print("   🧠 Consciousness: Integrated")
    print("🎮 Features:")
    print("   ✅ Grammar rule checking")
    print("   ✅ Style and readability analysis")
    print("   ✅ Harmonic pattern recognition")
    print("   ✅ Consciousness field integration")
    print("   ✅ Text improvement suggestions")
    print("   ✅ Modern web interface")
    print("=" * 50)

    try:
        with socketserver.TCPServer(("", port), GrammarHandler) as httpd:
            print(f"🚀 Grammar server started successfully on port {port}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    main()
