import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
// Import your domain-specific components
import {{SystemMetricsComponent}} from "@/components/{{SystemMetricsComponent}}";
import {{EntityCardComponent}} from "@/components/{{EntityCardComponent}}";
import {{JobCardComponent}} from "@/components/{{JobCardComponent}}";
import {{ChartComponent}} from "@/components/{{ChartComponent}}";
import {{AllocationComponent}} from "@/components/{{AllocationComponent}}";
import {{LogComponent}} from "@/components/{{LogComponent}}";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { {{primaryStatusIcon}}, {{secondaryStatusIcon}}, {{tertiaryStatusIcon}}, {{quaternaryStatusIcon}}, Loader2 } from "lucide-react";
import type { {{EntityResponseType}}, {{JobResponseType}}, {{SystemStatusType}} } from "@shared/schema";

// Helper function to format duration - customize for your domain
function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  } else {
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
}

export default function Dashboard() {
  // Fetch primary entities from API
  const {
    data: {{entities}} = [],
    isLoading: {{entities}}Loading,
    error: {{entities}}Error
  } = useQuery<{{EntityResponseType}}[]>({
    queryKey: ["/api/{{entities}}"],
    refetchInterval: {{entityRefreshInterval}}, // Refresh interval in ms
  });

  // Fetch jobs/tasks from API
  const {
    data: {{jobs}} = [],
    isLoading: {{jobs}}Loading,
    error: {{jobs}}Error
  } = useQuery<{{JobResponseType}}[]>({
    queryKey: ["/api/{{jobs}}"],
    refetchInterval: {{jobRefreshInterval}}, // Refresh interval in ms for job updates
  });

  // Fetch system status from API
  const {
    data: {{systemStatus}},
    isLoading: {{systemLoading}},
    error: {{systemError}}
  } = useQuery<{{SystemStatusType}}>({
    queryKey: ["/api/system/status"],
    refetchInterval: {{systemRefreshInterval}}, // Refresh interval in ms
  });

  // Generate performance data from system status - customize metrics
  const {{performanceData}} = {{systemStatus}} ? [
    {
      time: new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' }),
      {{primaryMetric}}: {{systemStatus}}.{{primaryMetric}}, // e.g., cpuUtilization
      {{secondaryMetric}}: {{systemStatus}}.{{secondaryMetric}}, // e.g., memoryUsage
      {{tertiaryMetric}}: {{systemStatus}}.{{tertiaryMetric}}, // e.g., throughput
      {{quaternaryMetric}}: {{systemStatus}}.{{quaternaryMetric}} // e.g., activeJobs
    }
  ] : [];

  // Fetch system logs from API
  const {
    data: {{logs}} = [],
    isLoading: {{logsLoading}},
    error: {{logsError}}
  } = useQuery<any[]>({
    queryKey: ["/api/{{logs}}"],
    refetchInterval: {{logsRefreshInterval}}, // Refresh interval in ms
  });

  // Show loading state
  if ({{entities}}Loading || {{jobs}}Loading || {{systemLoading}}) {
    return (
      <div className="flex items-center justify-center h-64" data-testid="loading-dashboard">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">{{loadingMessage}}</span>
      </div>
    );
  }

  // Show error state
  if ({{entities}}Error || {{jobs}}Error || {{systemError}}) {
    return (
      <div className="flex items-center justify-center h-64" data-testid="error-dashboard">
        <{{primaryStatusIcon}} className="h-8 w-8 text-red-500" />
        <span className="ml-2 text-red-500">
          Error loading dashboard data: {(vgpusError as any)?.message || (jobsError as any)?.message || (systemError as any)?.message}
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="dashboard-main">
      {/* System Status Header */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{{primaryStatusTitle}}</CardTitle>
            <{{primaryStatusIcon}} className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{{systemStatus}}?.{{primaryStatusValue}} || 0</div>
            <p className="text-xs text-muted-foreground">{{primaryStatusDescription}}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{{secondaryStatusTitle}}</CardTitle>
            <{{secondaryStatusIcon}} className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{{systemStatus}}?.{{secondaryStatusValue}} || 0</div>
            <p className="text-xs text-muted-foreground">{{secondaryStatusDescription}}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{{tertiaryStatusTitle}}</CardTitle>
            <{{tertiaryStatusIcon}} className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{{systemStatus}}?.{{tertiaryStatusValue}} || 0</div>
            <p className="text-xs text-muted-foreground">{{tertiaryStatusDescription}}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{{quaternaryStatusTitle}}</CardTitle>
            <{{quaternaryStatusIcon}} className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{{systemStatus}}?.{{quaternaryStatusValue}} || 0</div>
            <p className="text-xs text-muted-foreground">{{quaternaryStatusDescription}}</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* System Metrics */}
        <div className="md:col-span-2 lg:col-span-2">
          <{{SystemMetricsComponent}} />
        </div>

        {/* Performance Chart */}
        <div className="md:col-span-2 lg:col-span-1">
          <{{ChartComponent}}
            data={{{performanceData}}}
            title="{{chartTitle}}"
            data-testid="chart-performance"
          />
        </div>

        {/* Primary Entities */}
        <div className="md:col-span-2 lg:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">{{entitiesTitle}}</h2>
            <Badge variant="outline">{{entities}}.length</Badge>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {{{entities}}.map(({{entity}}) => (
              <{{EntityCardComponent}}
                key={{{entity}}.id}
                id={{{entity}}.id}
                name={{{entity}}.name}
                {{entityCardProps}}
                onToggle={async (id) => {
                  // Handle toggle action
                  try {
                    await apiRequest(`/api/{{entities}}/${id}/toggle`, {
                      method: 'POST',
                    });
                    queryClient.invalidateQueries({ queryKey: ["/api/{{entities}}"] });
                  } catch (error) {
                    console.error('Toggle failed:', error);
                  }
                }}
                onConfigure={(id) => {
                  // Handle configure action
                  console.log('Configure', id);
                }}
              />
            ))}
          </div>
        </div>

        {/* Jobs/Tasks */}
        <div className="md:col-span-2 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">{{jobsTitle}}</h2>
            <Badge variant="outline">{{jobs}}.length</Badge>
          </div>
          <div className="space-y-4">
            {{{jobs}}.slice(0, 5).map(({{job}}) => (
              <{{JobCardComponent}}
                key={{{job}}.id}
                id={{{job}}.id}
                name={{{job}}.name}
                {{jobCardProps}}
                onCancel={async (id) => {
                  // Handle cancel action
                  try {
                    await apiRequest(`/api/{{jobs}}/${id}/cancel`, {
                      method: 'POST',
                    });
                    queryClient.invalidateQueries({ queryKey: ["/api/{{jobs}}"] });
                  } catch (error) {
                    console.error('Cancel failed:', error);
                  }
                }}
              />
            ))}
          </div>
        </div>

        {/* Resource Allocation */}
        <div className="md:col-span-2 lg:col-span-1">
          <{{AllocationComponent}} />
        </div>

        {/* System Logs */}
        <div className="md:col-span-2 lg:col-span-3">
          <{{LogComponent}}
            logs={{{logs}}}
            title="{{logsTitle}}"
            data-testid="logs-system"
          />
        </div>
      </div>
    </div>
  );
}

/*
CUDNT TEMPLATE CONFIGURATION:

Replace these placeholders with your domain-specific values:

{{SystemMetricsComponent}}: System metrics component name
{{EntityCardComponent}}: Primary entity card component name
{{JobCardComponent}}: Job/task card component name
{{ChartComponent}}: Performance chart component name
{{AllocationComponent}}: Resource allocation component name
{{LogComponent}}: Log viewer component name

{{EntityResponseType}}: API response type for primary entities
{{JobResponseType}}: API response type for jobs/tasks
{{SystemStatusType}}: API response type for system status

{{entities}}: Plural name for primary entities (e.g., vgpus, models)
{{jobs}}: Plural name for jobs/tasks (e.g., jobs, tasks)
{{systemStatus}}: System status object name
{{logs}}: Logs array name

{{entityRefreshInterval}}: Refresh interval for entities (e.g., 5000)
{{jobRefreshInterval}}: Refresh interval for jobs (e.g., 2000)
{{systemRefreshInterval}}: Refresh interval for system status (e.g., 5000)
{{logsRefreshInterval}}: Refresh interval for logs (e.g., 10000)

{{performanceData}}: Performance data array name
{{primaryMetric}}: Primary metric field (e.g., cpuUtilization)
{{secondaryMetric}}: Secondary metric field (e.g., memoryUsage)
{{tertiaryMetric}}: Tertiary metric field (e.g., throughput)
{{quaternaryMetric}}: Quaternary metric field (e.g., activeJobs)

{{primaryStatusIcon}}: Icon for primary status card
{{secondaryStatusIcon}}: Icon for secondary status card
{{tertiaryStatusIcon}}: Icon for tertiary status card
{{quaternaryStatusIcon}}: Icon for quaternary status card

{{primaryStatusTitle}}: Title for primary status card
{{primaryStatusValue}}: Value field for primary status
{{primaryStatusDescription}}: Description for primary status

{{secondaryStatusTitle}}: Title for secondary status card
{{secondaryStatusValue}}: Value field for secondary status
{{secondaryStatusDescription}}: Description for secondary status

{{tertiaryStatusTitle}}: Title for tertiary status card
{{tertiaryStatusValue}}: Value field for tertiary status
{{tertiaryStatusDescription}}: Description for tertiary status

{{quaternaryStatusTitle}}: Title for quaternary status card
{{quaternaryStatusValue}}: Value field for quaternary status
{{quaternaryStatusDescription}}: Description for quaternary status

{{loadingMessage}}: Loading message text

{{entitiesTitle}}: Title for entities section
{{jobsTitle}}: Title for jobs section
{{chartTitle}}: Title for performance chart
{{logsTitle}}: Title for logs section

{{entity}}: Singular entity variable name
{{job}}: Singular job variable name

{{entityCardProps}}: Additional props for entity card
{{jobCardProps}}: Additional props for job card

EXAMPLE CONFIGURATION FOR CUDNT:
{{SystemMetricsComponent}}: SystemMetrics
{{EntityCardComponent}}: VirtualGPUCard
{{JobCardComponent}}: ComputeJobCard
{{ChartComponent}}: PerformanceChart
{{AllocationComponent}}: ResourceAllocation
{{LogComponent}}: LogViewer

{{EntityResponseType}}: VirtualGPUResponse
{{JobResponseType}}: ComputeJobResponse
{{SystemStatusType}}: SystemStatus

{{entities}}: vgpus
{{jobs}}: jobs
{{systemStatus}}: systemStatus
{{logs}}: logs

{{entityRefreshInterval}}: 5000
{{jobRefreshInterval}}: 2000
{{systemRefreshInterval}}: 5000
{{logsRefreshInterval}}: 10000

{{performanceData}}: performanceData
{{primaryMetric}}: cpuUtilization
{{secondaryMetric}}: memoryUtilization
{{tertiaryMetric}}: throughput
{{quaternaryMetric}}: runningJobs

{{primaryStatusIcon}}: Activity
{{secondaryStatusIcon}}: Cpu
{{tertiaryStatusIcon}}: Zap
{{quaternaryStatusIcon}}: CheckCircle

{{primaryStatusTitle}}: System Load
{{primaryStatusValue}}: cpuUtilization
{{primaryStatusDescription}}: Current CPU utilization

{{secondaryStatusTitle}}: Virtual GPUs
{{secondaryStatusValue}}: totalVgpus
{{secondaryStatusDescription}}: Total configured VGPUs

{{tertiaryStatusTitle}}: Active Jobs
{{tertiaryStatusValue}}: runningJobs
{{tertiaryStatusDescription}}: Currently running jobs

{{quaternaryStatusTitle}}: Throughput
{{quaternaryStatusValue}}: throughput
{{quaternaryStatusDescription}}: Operations per second

{{loadingMessage}}: Loading dashboard data...

{{entitiesTitle}}: Virtual GPUs
{{jobsTitle}}: Compute Jobs
{{chartTitle}}: Performance Metrics
{{logsTitle}}: System Logs

{{entity}}: vgpu
{{job}}: job

{{entityCardProps}}:
cores={vgpu.cores}
utilization={vgpu.utilization}
status={vgpu.status}
memory={vgpu.memory}

{{jobCardProps}}:
status={job.status}
progress={job.progress}
priority={job.priority}
duration={formatDuration(job.duration)}
*/
