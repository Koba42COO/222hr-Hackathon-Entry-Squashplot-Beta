import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";
// Import your specific icons - customize these based on your domain
import {
  {{primaryIcon}}, // e.g., Cpu, Brain, Database
  {{secondaryIcon}}, // e.g., Activity, TrendingUp
  {{settingsIcon}}, // e.g., Settings2
  {{systemIcon}}, // e.g., Server, Network
  Home,
  Zap,
  FileText,
  Database
} from "lucide-react";
import { Link, useLocation } from "wouter";

// Menu configuration - customize for your domain
const menuItems = [
  {
    title: "{{primarySection}}", // e.g., "Dashboard", "Compute", "Analysis"
    url: "/",
    icon: Home,
  },
  {
    title: "{{secondarySection}}", // e.g., "Virtual GPUs", "Models", "Datasets"
    url: "/{{secondaryRoute}}",
    icon: {{primaryIcon}},
  },
  {
    title: "{{tertiarySection}}", // e.g., "Compute Jobs", "Training", "Processing"
    url: "/{{tertiaryRoute}}",
    icon: Zap,
  },
  {
    title: "{{analyticsSection}}", // e.g., "Performance", "Analytics", "Metrics"
    url: "/{{analyticsRoute}}",
    icon: TrendingUp,
  },
  {
    title: "{{resourceSection}}", // e.g., "Resource Manager", "Settings", "Configuration"
    url: "/{{resourceRoute}}",
    icon: Settings2,
  },
  {
    title: "{{logsSection}}", // e.g., "System Logs", "Audit", "History"
    url: "/{{logsRoute}}",
    icon: FileText,
  },
  {
    title: "{{specializedSection}}", // e.g., "Wallace Transform", "Advanced Analysis"
    url: "/{{specializedRoute}}",
    icon: Brain,
  },
];

// System monitoring items - typically don't change
const systemItems = [
  {
    title: "System Status",
    url: "/status",
    icon: Activity,
  },
  {
    title: "{{systemBridge}}", // e.g., "Python Bridge", "API Gateway"
    url: "/bridge",
    icon: Database,
  },
];

export function AppSidebar() {
  const [location] = useLocation();

  return (
    <Sidebar data-testid="sidebar-main">
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded bg-primary">
            <{{primaryIcon}} className="h-4 w-4 text-primary-foreground" />
          </div>
          <div>
            <div className="text-sm font-semibold">{{brandName}}</div>
            <div className="text-xs text-muted-foreground">{{tagline}}</div>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>{{primaryGroupLabel}}</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url}
                    data-testid={`nav-${item.title.toLowerCase().replace(/\s+/g, '-')}`}
                  >
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>System</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {systemItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url}
                    data-testid={`nav-${item.title.toLowerCase().replace(/\s+/g, '-')}`}
                  >
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}

/*
CUDNT TEMPLATE CONFIGURATION:

Replace these placeholders with your domain-specific values:

{{primaryIcon}}: Main icon for your platform (Cpu, Brain, Database, etc.)
{{secondaryIcon}}: Secondary navigation icon (Activity, TrendingUp, etc.)
{{settingsIcon}}: Settings/configuration icon
{{systemIcon}}: System monitoring icon

{{primarySection}}: Main dashboard section name
{{secondarySection}}: Primary feature section name
{{tertiarySection}}: Secondary feature section name
{{analyticsSection}}: Analytics/monitoring section name
{{resourceSection}}: Resource management section name
{{logsSection}}: Logging/audit section name
{{specializedSection}}: Specialized/advanced feature section name

{{secondaryRoute}}: URL path for secondary section (without leading slash)
{{tertiaryRoute}}: URL path for tertiary section
{{analyticsRoute}}: URL path for analytics section
{{resourceRoute}}: URL path for resource section
{{logsRoute}}: URL path for logs section
{{specializedRoute}}: URL path for specialized section

{{brandName}}: Your platform name (e.g., "CUDNT Platform")
{{tagline}}: Short descriptive tagline (e.g., "Virtual GPU Manager")

{{primaryGroupLabel}}: Label for main navigation group (e.g., "Platform", "Compute")
{{systemBridge}}: Name for system bridge (e.g., "Python Bridge", "API Gateway")

EXAMPLE CONFIGURATION FOR CUDNT:
{{primaryIcon}}: Cpu
{{secondaryIcon}}: Activity
{{settingsIcon}}: Settings2
{{systemIcon}}: Database

{{primarySection}}: Dashboard
{{secondarySection}}: Virtual GPUs
{{tertiarySection}}: Compute Jobs
{{analyticsSection}}: Performance
{{resourceSection}}: Resource Manager
{{logsSection}}: System Logs
{{specializedSection}}: Wallace Transform

{{secondaryRoute}}: vgpus
{{tertiaryRoute}}: jobs
{{analyticsRoute}}: performance
{{resourceRoute}}: resources
{{logsRoute}}: logs
{{specializedRoute}}: wallace

{{brandName}}: CUDNT Platform
{{tagline}}: Virtual GPU Manager
{{primaryGroupLabel}}: Platform
{{systemBridge}}: Python Bridge
*/
