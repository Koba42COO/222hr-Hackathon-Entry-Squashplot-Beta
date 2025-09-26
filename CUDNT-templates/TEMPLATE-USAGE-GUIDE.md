# CUDNT Templating System Usage Guide

## Overview

This templating system captures the complete CUDNT platform architecture and design system, enabling you to quickly bootstrap enterprise-grade virtual GPU management platforms or adapt the patterns for other technical computing applications.

## Template Categories

### 1. Project Structure (`project-structure.json`)
**Purpose**: Complete project blueprint with all directories, files, and dependencies
**Usage**: Reference for understanding the full-stack architecture
**Contains**: 644+ files across frontend, backend, Python engine, and configuration

### 2. Component Templates (`components/`)
- **AppSidebar.template.tsx**: Technical navigation sidebar with grouped menu items
- **DataCard.template.tsx**: Data-dense cards for entity management (VGPU, jobs, etc.)
- **Dashboard.template.tsx**: Main dashboard layout with real-time metrics and grids

### 3. Styling Templates (`styling/`)
- **index.css.template**: Complete CSS variables system with light/dark themes
- **tailwind.config.ts.template**: Tailwind configuration with CUDNT design tokens

### 4. Build Templates (`build/`)
- **package.json.template**: Full dependency list and npm scripts
- **.replit.template**: Replit deployment configuration

### 5. Documentation (`documentation/`)
- **README.template.md**: Comprehensive project documentation template

## Quick Start: Creating a New CUDNT-Style Project

### Step 1: Initialize Project Structure
```bash
# Create new project directory
mkdir my-cudnt-project
cd my-cudnt-project

# Copy template files (replace placeholders with your values)
cp CUDNT-templates/project-structure.json ./project-structure.json
# Follow the structure to create directories and files
```

### Step 2: Configure Core Templates

#### AppSidebar Configuration (Example: AI Model Manager)
```typescript
// Replace in AppSidebar.template.tsx:
{{primaryIcon}}: Brain
{{secondaryIcon}}: TrendingUp
{{brandName}}: AI Model Manager
{{tagline}}: Neural Network Orchestrator

{{primarySection}}: Dashboard
{{secondarySection}}: Models
{{tertiarySection}}: Training Jobs
{{analyticsSection}}: Performance
{{resourceSection}}: Resource Manager
{{logsSection}}: System Logs
{{specializedSection}}: Model Registry

{{secondaryRoute}}: models
{{tertiaryRoute}}: jobs
{{analyticsRoute}}: performance
{{resourceRoute}}: resources
{{logsRoute}}: logs
{{specializedRoute}}: registry
```

#### DataCard Configuration (Example: ML Model Card)
```typescript
// Replace in DataCard.template.tsx:
{{EntityName}}: MLModel
{{entityName}}: ml-model

{{primaryIcon}}: Brain
{{actionIcon}}: Play
{{configIcon}}: Settings

{{primaryMetric}}: parameters
{{primaryMetricType}}: number
{{secondaryMetric}}: accuracy
{{secondaryMetricType}}: number

{{statusType}}: "training" | "ready" | "failed" | "stopped"

{{resourceMetric}}: memory

{{activeStatus}}: training
{{idleStatus}}: ready
{{errorStatus}}: failed
{{disabledStatus}}: stopped

{{primaryMetricLabel}}: Parameters
{{primaryMetricDisplay}}: {parameters}M
{{secondaryMetricLabel}}: Accuracy
{{secondaryMetricDisplay}}: {accuracy}%

{{primaryProgressLabel}}: Training Progress
{{primaryProgressValue}}: progress
{{primaryProgressId}}: progress

{{resourceProgressLabel}}: Memory Usage
{{resourceProgressValue}}: {memory.used}GB / {memory.total}GB
{{resourceProgressId}}: memory

{{toggleAction}}: train
{{configAction}}: configure

{{toggleButtonText}}: {status === "training" ? "Stop" : "Train"}
{{configButtonText}}: Configure
```

#### Styling Configuration (Example: Data Science Theme)
```css
/* Replace in index.css.template: */
--background: 210 15% 6%;        /* Dark slate background */
--foreground: 210 10% 95%;       /* Light text */
--primary: 142 76% 45%;          /* Forest green for ML */
--chart-1: 142 76% 55%;          /* Success green */
--chart-2: 38 92% 65%;           /* Warning amber */
--chart-3: 210 100% 70%;         /* Computing blue */
--chart-4: 270 95% 75%;          /* Data purple */
--chart-5: 20 90% 70%;           /* Training orange */

--font-sans: Inter, sans-serif;   /* Clean, readable */
--font-mono: JetBrains Mono, monospace; /* Code-friendly */
```

### Step 3: Package.json Configuration
```json
{
  "name": "ai-model-manager",
  "version": "1.0.0",
  "scripts": {
    "dev": "NODE_ENV=development tsx server/index.ts",
    "build": "vite build && esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist",
    "start": "NODE_ENV=production node dist/index.js"
  }
}
```

### Step 4: Replit Configuration
```toml
modules = ["nodejs-20", "web", "postgresql-16", "python-3.11"]
run = "npm run dev"
hidden = [".config", ".git", "node_modules", "dist"]

[deployment]
deploymentTarget = "autoscale"
build = ["npm", "run", "build"]
run = ["npm", "run", "start"]

[env]
PORT = "5000"
```

## Design System Customization

### Color Schemes
The template supports complete color customization through CSS variables:

#### Technical Computing Theme (CUDNT Default)
```css
--primary: 210 100% 60%;     /* Electric blue */
--background: 220 15% 8%;     /* Deep slate */
--success: 142 76% 36%;       /* Vibrant green */
--warning: 38 92% 50%;        /* Amber */
--error: 0 84% 60%;          /* Red */
```

#### Data Science Theme
```css
--primary: 142 76% 45%;      /* Forest green */
--background: 210 15% 6%;     /* Dark slate */
--success: 142 76% 55%;       /* Success green */
--warning: 38 92% 65%;        /* Training amber */
--error: 0 84% 55%;          /* Error red */
```

#### Cloud Infrastructure Theme
```css
--primary: 270 95% 60%;      /* Infrastructure purple */
--background: 220 15% 10%;    /* Cloud dark */
--success: 142 76% 50%;       /* Health green */
--warning: 38 92% 60%;        /* Alert amber */
--error: 0 84% 55%;          /* Critical red */
```

### Component Patterns

#### Navigation Patterns
- **Technical Dashboard**: Hierarchical navigation (Platform → System)
- **Resource Management**: CRUD-focused with real-time updates
- **Analytics**: Data-heavy with multiple chart types

#### Layout Patterns
- **Sidebar + Main**: Technical navigation with content area
- **Grid-based**: Responsive card layouts for data density
- **Real-time Updates**: WebSocket-powered live data

#### Data Visualization
- **Performance Charts**: Line/area charts for metrics
- **Status Indicators**: Color-coded badges and progress bars
- **Log Viewers**: Syntax-highlighted, scrollable terminals

## Architecture Patterns

### Full-Stack Architecture
```
Frontend (React + TypeScript)
├── Components (shadcn/ui)
├── State (TanStack Query)
├── Routing (Wouter)
└── Styling (Tailwind + CSS Variables)

Backend (Node.js + TypeScript)
├── API (Express + REST)
├── WebSocket (Real-time)
├── Database (Drizzle + PostgreSQL)
└── Validation (Zod)

Python Engine
├── Compute Operations
├── Bridge API (WebSocket)
└── Resource Management
```

### Real-time Data Flow
```
Python Engine → WebSocket → Backend → TanStack Query → React Components
                      ↑                    ↓
                Live Updates      Automatic Cache Invalidation
```

### Database Schema Pattern
```typescript
// Type-safe schemas with Drizzle
export const entities = pgTable("entities", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  status: text("status").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});
```

## Deployment Patterns

### Replit Autoscale
```toml
[deployment]
deploymentTarget = "autoscale"
build = ["npm", "run", "build"]
run = ["npm", "run", "start"]
```

### Development Workflow
```bash
npm run dev          # Hot-reload development
npm run check        # TypeScript validation
npm run build        # Production build
npm run start        # Production server
```

## Best Practices

### Code Organization
- **Components**: Feature-based organization
- **Types**: Shared schemas for type safety
- **API**: RESTful endpoints with Zod validation
- **Styling**: CSS variables for theming

### Performance
- **Real-time**: WebSocket for live updates
- **Caching**: TanStack Query for state management
- **Bundling**: Vite for fast development builds

### Testing
- **Components**: React Testing Library
- **API**: Integration tests
- **E2E**: Cypress for user flows

## Example Implementations

### 1. Virtual GPU Manager (CUDNT Original)
- GPU simulation with CPU cores
- Real-time performance monitoring
- Job scheduling and resource allocation

### 2. AI Model Training Platform
- Model registry and versioning
- Training job management
- Performance analytics and metrics

### 3. Cloud Resource Orchestrator
- Multi-cloud resource management
- Cost optimization
- Real-time monitoring dashboard

### 4. Scientific Computing Platform
- Parallel computation scheduling
- Data pipeline management
- Research collaboration tools

## Template Variables Reference

### Common Replacements
- `{{projectName}}`: Your project name
- `{{EntityName}}`: PascalCase entity name
- `{{entityName}}`: kebab-case entity name
- `{{primaryIcon}}`: Main Lucide icon
- `{{primaryColor}}`: HSL color value

### Domain-Specific Variables
- `{{primaryEntity}}`: Main business entity
- `{{secondaryEntity}}`: Secondary business entity
- `{{performanceMetric}}`: Key performance indicator
- `{{realTimeData}}`: Live data type

## Support and Extensions

The templating system is designed to be:
- **Modular**: Mix and match components
- **Extensible**: Add new patterns easily
- **Type-safe**: Full TypeScript integration
- **Production-ready**: Optimized for deployment

For questions or contributions, refer to the individual template files for detailed configuration options.
