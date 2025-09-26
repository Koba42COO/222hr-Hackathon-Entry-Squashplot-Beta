import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
// Import domain-specific icons
import { {{primaryIcon}}, {{actionIcon}}, {{configIcon}} } from "lucide-react";

interface {{EntityName}}CardProps {
  id: string;
  name: string;
  // Primary metrics - customize based on your domain
  {{primaryMetric}}: {{primaryMetricType}}; // e.g., cores: number, status: string
  {{secondaryMetric}}: {{secondaryMetricType}}; // e.g., utilization: number

  // Status and state
  status: {{statusType}}; // e.g., "active" | "idle" | "error" | "disabled"

  // Resource metrics - customize for your domain
  {{resourceMetric}}: {
    used: number;
    total: number;
  };

  // Action handlers
  onToggle: (id: string) => void;
  onConfigure: (id: string) => void;
}

export default function {{EntityName}}Card({
  id,
  name,
  {{primaryMetric}},
  {{secondaryMetric}},
  status,
  {{resourceMetric}},
  onToggle,
  onConfigure,
}: {{EntityName}}CardProps) {
  // Status color mapping - customize for your domain states
  const getStatusColor = (status: string) => {
    switch (status) {
      case "{{activeStatus}}": return "bg-chart-1"; // e.g., "active", "running"
      case "{{idleStatus}}": return "bg-muted"; // e.g., "idle", "stopped"
      case "{{errorStatus}}": return "bg-destructive"; // e.g., "error", "failed"
      case "{{disabledStatus}}": return "bg-muted-foreground"; // e.g., "disabled", "offline"
      default: return "bg-muted";
    }
  };

  // Calculate resource percentages
  const {{resourceMetric}}Percent = {{resourceMetric}}.total > 0
    ? ({{resourceMetric}}.used / {{resourceMetric}}.total) * 100
    : 0;

  return (
    <Card className="hover-elevate" data-testid={`card-{{entityName}}-${id}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <{{primaryIcon}} className="h-4 w-4" />
          {name}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={`${getStatusColor(status)} border-0 text-white`}
            data-testid={`status-${id}`}
          >
            {status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {/* Primary Metrics Grid */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">{{primaryMetricLabel}}</div>
              <div className="font-mono font-medium">{{primaryMetricDisplay}}</div>
            </div>
            <div>
              <div className="text-muted-foreground">{{secondaryMetricLabel}}</div>
              <div className="font-mono font-medium">{{secondaryMetricDisplay}}</div>
            </div>
          </div>

          {/* Primary Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">{{primaryProgressLabel}}</span>
              <span className="font-mono">{{primaryProgressValue}}</span>
            </div>
            <Progress value={{{primaryProgressValue}}} className="h-2" data-testid={`progress-{{primaryProgressId}}-${id}`} />
          </div>

          {/* Resource Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">{{resourceProgressLabel}}</span>
              <span className="font-mono">{{resourceProgressValue}}</span>
            </div>
            <Progress value={{{resourceMetric}}Percent} className="h-2" data-testid={`progress-{{resourceProgressId}}-${id}`} />
          </div>

          {/* Action Buttons */}
          <div className="flex justify-between gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onToggle(id)}
              data-testid={`button-{{toggleAction}}-${id}`}
            >
              <{{actionIcon}} className="h-3 w-3 mr-1" />
              {{{toggleButtonText}}}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onConfigure(id)}
              data-testid={`button-{{configAction}}-${id}`}
            >
              <{{configIcon}} className="h-3 w-3 mr-1" />
              {{configButtonText}}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/*
CUDNT TEMPLATE CONFIGURATION:

Replace these placeholders with your domain-specific values:

{{EntityName}}: Name of your entity (e.g., VirtualGPU, ComputeJob, Model)
{{entityName}}: Lowercase version for test IDs (e.g., virtual-gpu, compute-job)

{{primaryIcon}}: Main icon for the card (e.g., Cpu, Zap, Brain)
{{actionIcon}}: Action button icon (e.g., Power, Play, Pause)
{{configIcon}}: Configuration button icon (e.g., Settings)

{{primaryMetric}}: Primary metric field name (e.g., cores, priority)
{{primaryMetricType}}: Type of primary metric (e.g., number, string)
{{secondaryMetric}}: Secondary metric field name (e.g., utilization, progress)
{{secondaryMetricType}}: Type of secondary metric (e.g., number)

{{statusType}}: Union type for status values (e.g., "active" | "idle" | "error" | "disabled")

{{resourceMetric}}: Resource metric field name (e.g., memory, storage, bandwidth)

{{activeStatus}}: Active/running status value (e.g., "active", "running")
{{idleStatus}}: Idle/stopped status value (e.g., "idle", "stopped")
{{errorStatus}}: Error/failed status value (e.g., "error", "failed")
{{disabledStatus}}: Disabled/offline status value (e.g., "disabled", "offline")

{{primaryMetricLabel}}: Display label for primary metric (e.g., "CPU Cores", "Priority")
{{primaryMetricDisplay}}: How to display primary metric (e.g., {cores}, {priority})
{{secondaryMetricLabel}}: Display label for secondary metric (e.g., "Utilization", "Progress")
{{secondaryMetricDisplay}}: How to display secondary metric (e.g., {utilization}%, {progress}%)

{{primaryProgressLabel}}: Label for primary progress bar (e.g., "CPU Utilization", "Training Progress")
{{primaryProgressValue}}: Value for primary progress bar (e.g., utilization, progress)
{{primaryProgressId}}: Test ID suffix for primary progress (e.g., utilization, progress)

{{resourceProgressLabel}}: Label for resource progress bar (e.g., "Memory", "Storage", "Bandwidth")
{{resourceProgressValue}}: Display value for resource progress (e.g., {memory.used}GB / {memory.total}GB)
{{resourceProgressId}}: Test ID suffix for resource progress (e.g., memory, storage)

{{toggleAction}}: Action name for toggle button test ID (e.g., toggle, start, stop)
{{configAction}}: Action name for config button test ID (e.g., configure, settings, edit)

{{toggleButtonText}}: Dynamic button text (e.g., {status === "active" ? "Stop" : "Start"})
{{configButtonText}}: Configuration button text (e.g., "Configure", "Settings", "Edit")

EXAMPLE CONFIGURATION FOR VIRTUAL GPU CARD:
{{EntityName}}: VirtualGPU
{{entityName}}: virtual-gpu

{{primaryIcon}}: Cpu
{{actionIcon}}: Power
{{configIcon}}: Settings

{{primaryMetric}}: cores
{{primaryMetricType}}: number
{{secondaryMetric}}: utilization
{{secondaryMetricType}}: number

{{statusType}}: "active" | "idle" | "error" | "disabled"

{{resourceMetric}}: memory

{{activeStatus}}: active
{{idleStatus}}: idle
{{errorStatus}}: error
{{disabledStatus}}: disabled

{{primaryMetricLabel}}: CPU Cores
{{primaryMetricDisplay}}: {cores}
{{secondaryMetricLabel}}: Utilization
{{secondaryMetricDisplay}}: {utilization}%

{{primaryProgressLabel}}: CPU Utilization
{{primaryProgressValue}}: utilization
{{primaryProgressId}}: utilization

{{resourceProgressLabel}}: Memory
{{resourceProgressValue}}: {memory.used}GB / {memory.total}GB
{{resourceProgressId}}: memory

{{toggleAction}}: toggle
{{configAction}}: configure

{{toggleButtonText}}: {status === "active" ? "Stop" : "Start"}
{{configButtonText}}: Configure
*/
