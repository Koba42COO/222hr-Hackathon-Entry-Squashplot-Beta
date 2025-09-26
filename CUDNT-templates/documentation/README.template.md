# {{projectName}} - {{projectTagline}}

## Overview

{{projectName}} is a {{projectType}} platform that {{projectDescription}}. The system provides enterprise-grade {{primaryCapability}} capabilities, enabling users to create, configure, and monitor {{primaryEntities}} for {{primaryUseCase}}. The platform bridges {{bridgeDescription}}, offering real-time {{realTimeFeatures}} with mathematical optimization through {{optimizationAlgorithm}} algorithms.

## User Preferences

Preferred communication style: {{communicationStyle}}

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript running on Vite for fast development and building
- **UI Library**: Shadcn/ui components built on Radix UI primitives with Tailwind CSS for styling
- **Design System**: Enterprise computing platform aesthetic inspired by {{designInspirations}}
- **Color Scheme**: {{colorSchemeDescription}}
- **State Management**: TanStack Query for server state management and caching
- **Routing**: Wouter for lightweight client-side routing

### Backend Architecture
- **Runtime**: Node.js with Express.js REST API server
- **Language**: TypeScript with strict type checking for reliability
- **Database Integration**: Drizzle ORM with PostgreSQL (Neon serverless) for data persistence
- **Real-time Communication**: WebSocket server for {{realTimeBridge}} integration
- **Session Management**: PostgreSQL-based session storage using connect-pg-simple

### Data Storage Solutions
- **Primary Database**: PostgreSQL via Neon serverless platform
- **ORM**: Drizzle ORM with type-safe schema definitions
- **Schema Structure**:
  - `{{primaryEntityTable}}`: {{primaryEntityDescription}}
  - `{{secondaryEntityTable}}`: {{secondaryEntityDescription}}
  - `{{tertiaryEntityTable}}`: {{tertiaryEntityDescription}}
  - `{{performanceTable}}`: Real-time {{performanceMetrics}} collection
  - `{{logsTable}}`: Comprehensive system logging and audit trails

### Authentication and Authorization
- Session-based authentication using PostgreSQL session store
- Role-based access control for {{rbacScope}} operations
- Secure WebSocket connections for {{secureConnections}} communication

### {{specializedEngine}} Integration
- **Engine**: Custom {{engineType}} using {{processingMethod}}
- **Optimization**: {{optimizationName}} algorithms for {{performanceImprovement}} performance improvements
- **Bridge API**: WebSocket-based communication between {{bridgeComponents}}
- **Operations**: {{supportedOperations}}
- **Resource Management**: {{resourceAllocation}} and task scheduling

### Real-time Performance Monitoring
- WebSocket-based real-time metrics streaming
- Performance charts using Recharts for {{chartMetrics}}
- System status indicators with color-coded alerts
- Log viewer with filtering capabilities and export functionality

### {{jobScheduling}} System
- Priority-based job queuing ({{priorityLevels}})
- Automatic job assignment to available {{jobAssignmentTargets}}
- Progress tracking and status management
- Resource validation before job execution

### API Design Patterns
- RESTful API endpoints for CRUD operations on {{apiEntities}}
- Consistent error handling with Zod schema validation
- Query parameter validation for filtering and searching
- Atomic database operations for {{atomicOperations}} consistency

## External Dependencies

### Database Services
- **{{databaseProvider}}**: Serverless PostgreSQL hosting platform
- **Connection**: @neondatabase/serverless driver with WebSocket support

### UI Framework Dependencies
- **Radix UI**: Accessible component primitives (@radix-ui/react-*)
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Lucide React**: Icon library for consistent iconography
- **Recharts**: Chart library for {{chartPurpose}}

### Development Tools
- **Vite**: Fast build tool and development server
- **TypeScript**: Static type checking across frontend and backend
- **Drizzle Kit**: Database schema management and migrations
- **ESBuild**: Fast JavaScript bundler for production builds

### {{specializedLibrary}} Computing
- **{{coreLibrary}}**: Core {{libraryPurpose}} library for {{libraryOperations}}
- **{{monitoringLibrary}}**: System and process monitoring utilities
- **WebSockets**: Real-time communication with {{websocketTarget}} backend
- **{{parallelLibrary}}**: {{parallelPurpose}} simulation

### Session and Security
- **connect-pg-simple**: PostgreSQL session store for Express.js
- **Express Session**: Session management middleware
- **CORS**: Cross-origin resource sharing configuration

### Query and State Management
- **TanStack Query**: Server state management with automatic caching and synchronization
- **React Hook Form**: Form validation and submission handling
- **Hookform Resolvers**: Zod integration for form validation

## Design System Configuration

### Color Palette

#### Dark Mode (Primary)
- **Background**: {{darkBackgroundHsl}} ({{darkBackgroundDescription}})
- **Surface**: {{darkSurfaceHsl}} ({{darkSurfaceDescription}})
- **Primary**: {{primaryHsl}} ({{primaryDescription}})
- **Success**: {{successHsl}} ({{successDescription}})
- **Warning**: {{warningHsl}} ({{warningDescription}})
- **Critical**: {{criticalHsl}} ({{criticalDescription}})

#### Light Mode (Secondary)
- **Background**: {{lightBackgroundHsl}} ({{lightBackgroundDescription}})
- **Surface**: {{lightSurfaceHsl}} ({{lightSurfaceDescription}})
- **Text**: {{lightTextHsl}} ({{lightTextDescription}})

### Typography
- **Primary**: {{primaryFont}} ({{primaryFontPurpose}})
- **Monospace**: {{monoFont}} ({{monoFontPurpose}})
- **Spacing**: {{typographySpacing}} for consistent, technical precision

### Layout System
Tailwind spacing units: **{{layoutSpacing}}** for consistent, technical precision
- Dense information spacing ({{denseSpacing}})
- Clear section separation ({{sectionSpacing}})
- Major layout breaks ({{majorSpacing}})

### Component Library

#### Core Navigation
- **Sidebar**: Persistent {{sidebarPurpose}} navigation ({{sidebarItems}})
- **Top Bar**: {{topBarContent}}
- **Breadcrumbs**: Deep navigation within {{breadcrumbHierarchy}} hierarchies

#### Data Displays
- **Metrics Cards**: Real-time {{metricsCards}} tracking
- **Performance Graphs**: {{performanceGraphs}} for {{graphMetrics}}
- **Resource Tables**: Sortable, filterable lists of {{tableEntities}}
- **Log Viewers**: Monospace, scrollable, with syntax highlighting

#### Control Interfaces
- **{{controlType1}}**: {{controlDescription1}}
- **{{controlType2}}**: {{controlDescription2}}
- **Action Buttons**: Primary ({{primaryAction}}), secondary ({{secondaryAction}}), danger ({{dangerAction}})

#### Status Indicators
- **Health Badges**: Color-coded status ({{statusTypes}})
- **Progress Bars**: {{progressBars}} completion, {{progressMetrics}}
- **Live Indicators**: Pulsing dots for {{liveIndicators}}

### Visual Treatments

#### Gradients
- **{{gradient1}}**: {{gradient1Hsl}} to {{gradient1Hsl2}} ({{gradient1Purpose}})
- **{{gradient2}}**: {{gradient2Hsl}} to {{gradient2Hsl2}} ({{gradient2Purpose}})

#### Background Treatments
- **{{backgroundPattern}}**: {{backgroundPatternDescription}}
- **Panel depth**: Subtle shadows and borders for {{panelDepthPurpose}}
- **{{resourceZones}}**: Color-coded background tints for different {{zoneTypes}}

### Images
**{{imagePhilosophy}}** - This is a {{imageContext}} focused on {{imageFocus}} rather than {{imageAvoid}}. Any imagery should be:
- **{{imageType1}}**: {{imageDescription1}}
- **{{imageType2}}**: {{imageDescription2}}
- **{{imageType3}}**: {{imageDescription3}}

The design prioritizes {{designPriority1}} and {{designPriority2}}, appropriate for a professional {{platformType}} platform.

## Getting Started

### Prerequisites
- Node.js {{nodeVersion}}
- PostgreSQL {{postgresVersion}}
- Python {{pythonVersion}}

### Installation

1. **Clone the repository**
   ```bash
   git clone {{repositoryUrl}}
   cd {{projectName}}
   ```

2. **Install dependencies**
   ```bash
   npm install
   pip install -r python_engine/requirements.txt
   ```

3. **Database setup**
   ```bash
   npm run db:push
   ```

4. **Environment configuration**
   ```bash
   # Configure your .env file with database credentials
   cp .env.example .env
   ```

5. **Start development server**
   ```bash
   npm run dev
   ```

### Build for Production

```bash
npm run build
npm run start
```

## Project Structure

```
{{projectName}}/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/         # Route components
│   │   ├── hooks/         # Custom React hooks
│   │   └── lib/           # Utilities and configurations
├── server/                 # Express backend
│   ├── routes.ts          # API routes
│   ├── db.ts             # Database connections
│   └── storage.ts        # Data persistence
├── python_engine/         # {{pythonEnginePurpose}}
├── shared/                # TypeScript schemas
└── attached_assets/       # Documentation and assets
```

## API Documentation

### REST Endpoints

#### {{primaryEntityPlural}}
- `GET /api/{{primaryEntityPlural}}` - List all {{primaryEntityPlural}}
- `POST /api/{{primaryEntityPlural}}` - Create new {{primaryEntitySingle}}
- `GET /api/{{primaryEntityPlural}}/:id` - Get {{primaryEntitySingle}} by ID
- `PUT /api/{{primaryEntityPlural}}/:id` - Update {{primaryEntitySingle}}
- `DELETE /api/{{primaryEntityPlural}}/:id` - Delete {{primaryEntitySingle}}

#### {{secondaryEntityPlural}}
- `GET /api/{{secondaryEntityPlural}}` - List all {{secondaryEntityPlural}}
- `POST /api/{{secondaryEntityPlural}}/:id/{{action}}` - Perform action on {{secondaryEntitySingle}}

#### System
- `GET /api/system/status` - Get system status and metrics
- `GET /api/logs` - Get system logs

### WebSocket Events

#### {{pythonEngineName}} Bridge
- `{{bridgeConnectEvent}}` - Establish bridge connection
- `{{bridgeDataEvent}}` - Receive {{bridgeDataType}} data
- `{{bridgeControlEvent}}` - Send control commands

## Development

### Code Style
- **TypeScript**: Strict type checking enabled
- **ESLint**: Airbnb configuration
- **Prettier**: Consistent code formatting
- **Husky**: Pre-commit hooks

### Testing
- **Unit Tests**: Jest with React Testing Library
- **Integration Tests**: Cypress for E2E testing
- **API Tests**: Postman/Newman collections

### Deployment
- **Development**: Local development with hot reload
- **Staging**: Automated deployment on {{stagingPlatform}}
- **Production**: {{productionDeployment}} with monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

{{license}} - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- {{acknowledgment1}}
- {{acknowledgment2}}
- {{acknowledgment3}}
