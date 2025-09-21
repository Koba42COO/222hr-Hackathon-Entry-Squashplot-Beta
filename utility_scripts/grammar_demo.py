#!/usr/bin/env python3
from grammar_analyzer import GrammarAnalyzer

def main():
    print("🧠 CONSCIOUSNESS GRAMMAR ANALYZER DEMO")
    analyzer = GrammarAnalyzer()

    test_text = 'Consciousness is the most profound mystery in science.'
    analysis = analyzer.analyze_text(test_text)

    print(f"Overall Score: {analysis['overall_score']:.1f}")
    print(f"Grammar Issues: {sum(len(issues) for issues in analysis['grammar_issues'].values())}")

    if analysis['consciousness_metrics']:
        print(f"Meta Entropy: {analysis['consciousness_metrics']['meta_entropy']:.3f}")

    print("✅ Demo complete!")

if __name__ == "__main__":
    main()