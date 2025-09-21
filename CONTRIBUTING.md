# Contributing to SquashPlot Beta

Thank you for your interest in contributing to SquashPlot! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Basic understanding of Chia farming
- Familiarity with Flask/web development (for UI contributions)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/222hr-Hackathon-Entry-Squashplot-Beta.git
   cd 222hr-Hackathon-Entry-Squashplot-Beta
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run the development server**
   ```bash
   python main.py --web --port 5000
   ```

## 📝 How to Contribute

### Types of Contributions

- **Bug Reports**: Report issues and bugs
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit code improvements
- **Documentation**: Improve documentation
- **Testing**: Add or improve tests
- **UI/UX**: Improve the web interface

### Contribution Process

1. **Create an Issue**: Describe what you want to contribute
2. **Fork the Repository**: Create your own fork
3. **Create a Branch**: `git checkout -b feature/your-feature-name`
4. **Make Changes**: Implement your changes
5. **Test**: Ensure your changes work correctly
6. **Commit**: Use clear, descriptive commit messages
7. **Push**: Push to your fork
8. **Pull Request**: Create a pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions small and focused
- Add comments for complex logic

### Commit Messages

Use clear, descriptive commit messages:
```
feat: add compression level 5 support
fix: resolve memory leak in job queue
docs: update installation instructions
test: add unit tests for compression engine
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_compression.py

# Run with coverage
python -m pytest --cov=src
```

### Writing Tests
- Test files should be in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

## 📚 Documentation

### Code Documentation
- Use docstrings for all functions and classes
- Follow Google docstring format
- Include type hints where possible

### User Documentation
- Update README.md for user-facing changes
- Add examples for new features
- Keep installation instructions current

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Logs**: Relevant error messages or logs

## 💡 Feature Requests

When requesting features, please include:

1. **Description**: Clear description of the feature
2. **Use Case**: Why this feature would be useful
3. **Proposed Solution**: How you think it should work
4. **Alternatives**: Other solutions you've considered

## 🔒 Security

If you discover a security vulnerability, please:

1. **Do NOT** create a public issue
2. Email security@squashplot.dev
3. Include detailed information about the vulnerability
4. Allow time for the issue to be addressed before disclosure

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows the style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Clear commit messages

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] Manual testing completed
- [ ] No regressions

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

## 🏷️ Release Process

1. **Version Bumping**: Update version in `__init__.py`
2. **Changelog**: Update CHANGELOG.md
3. **Tagging**: Create a git tag for the release
4. **Documentation**: Update documentation if needed

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the code of conduct

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: dev@squashplot.dev for private matters

## 🎯 Hackathon Context

This project was developed for the 222hr Hackathon. Contributions should:

- Maintain the hackathon spirit of innovation
- Focus on practical, usable features
- Consider the Chia farming community's needs
- Prioritize performance and efficiency

Thank you for contributing to SquashPlot! 🚀
