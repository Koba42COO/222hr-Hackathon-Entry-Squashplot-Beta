import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, throwError, BehaviorSubject } from 'rxjs';
import { catchError, tap, retry } from 'rxjs/operators';

import { environment } from '../../../environments/environment';

export interface ApiResponse<T = any> {
  data?: T;
  result?: T;
  message?: string;
  status?: string;
  timestamp?: string;
  error?: string;
}

export interface ConsciousnessRequest {
  algorithm: string;
  parameters: {
    iterations?: number;
    phi?: number;
    sigma?: number;
    [key: string]: any;
  };
  input_data: number[];
}

export interface ConsciousnessResponse {
  performance_gain: number;
  correlation: number;
  processing_time: number;
  status: string;
  result?: any;
  consciousness_gain?: number;
}

export interface ChatMessage {
  content: string;
  provider?: 'openai' | 'anthropic' | 'google';
  timestamp: Date;
  userId: string;
  isBot?: boolean;
  metadata?: any;
}

export interface ChatResponse {
  content: string;
  provider: string;
  timestamp: Date;
  conversational_response?: string;
}

export interface QuantumSimulationConfig {
  qubits: number;
  iterations: number;
  algorithm: string;
}

export interface SystemMetrics {
  apiStatus: boolean;
  memoryUsage: number;
  activeSessions: number;
  performance: {
    responseTime: number;
    throughput: number;
  };
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiUrl;
  
  // Loading state management
  private loadingSubject = new BehaviorSubject<boolean>(false);
  public loading$ = this.loadingSubject.asObservable();

  // Error state management
  private errorSubject = new BehaviorSubject<string | null>(null);
  public error$ = this.errorSubject.asObservable();

  constructor() {
    console.log('🔗 ApiService initialized with base URL:', this.baseUrl);
  }

  // Generic HTTP methods
  get<T>(endpoint: string, params?: HttpParams): Observable<T> {
    this.setLoading(true);
    this.clearError();

    return this.http.get<T>(`${this.baseUrl}${endpoint}`, { params }).pipe(
      retry(2),
      tap(() => this.setLoading(false)),
      catchError(error => this.handleError(error))
    );
  }

  post<T>(endpoint: string, data: any, options?: { headers?: HttpHeaders }): Observable<T> {
    this.setLoading(true);
    this.clearError();

    return this.http.post<T>(`${this.baseUrl}${endpoint}`, data, options).pipe(
      retry(1),
      tap(() => this.setLoading(false)),
      catchError(error => this.handleError(error))
    );
  }

  put<T>(endpoint: string, data: any): Observable<T> {
    this.setLoading(true);
    this.clearError();

    return this.http.put<T>(`${this.baseUrl}${endpoint}`, data).pipe(
      retry(1),
      tap(() => this.setLoading(false)),
      catchError(error => this.handleError(error))
    );
  }

  delete<T>(endpoint: string): Observable<T> {
    this.setLoading(true);
    this.clearError();

    return this.http.delete<T>(`${this.baseUrl}${endpoint}`).pipe(
      retry(1),
      tap(() => this.setLoading(false)),
      catchError(error => this.handleError(error))
    );
  }

  // Specialized API methods for chAIos

  /**
   * Health check endpoint
   */
  healthCheck(): Observable<any> {
    return this.get('/health');
  }

  /**
   * System information
   */
  getSystemInfo(): Observable<any> {
    return this.get('/system/info');
  }

  /**
   * System metrics
   */
  getSystemMetrics(): Observable<SystemMetrics> {
    return this.get<SystemMetrics>('/system/metrics');
  }

  /**
   * Process consciousness data
   */
  processConsciousness(data: ConsciousnessRequest): Observable<ConsciousnessResponse> {
    console.log('🧠 Processing consciousness data:', data);
    return this.post<ConsciousnessResponse>('/consciousness/process', data);
  }

  /**
   * Get consciousness processing history
   */
  getConsciousnessHistory(): Observable<any[]> {
    return this.get<any[]>('/consciousness/history');
  }

  /**
   * Get real-time consciousness metrics
   */
  getConsciousnessMetrics(): Observable<any> {
    return this.get('/consciousness/metrics');
  }

  /**
   * Send chat message to AI
   */
  sendChatMessage(message: ChatMessage): Observable<ChatResponse> {
    console.log('💬 Sending chat message:', message);
    return this.post<ChatResponse>('/chat/message', {
      message: message.content,
      provider: message.provider || 'openai',
      userId: message.userId,
      timestamp: message.timestamp.toISOString()
    });
  }

  /**
   * Get chat history
   */
  getChatHistory(userId: string, limit: number = 50): Observable<ChatMessage[]> {
    const params = new HttpParams()
      .set('userId', userId)
      .set('limit', limit.toString());
    
    return this.get<ChatMessage[]>('/chat/history', params);
  }

  /**
   * Run quantum simulation
   */
  runQuantumSimulation(config: QuantumSimulationConfig): Observable<any> {
    console.log('⚛️ Running quantum simulation:', config);
    return this.post('/quantum-annealing', {
      algorithm: 'quantum_consciousness',
      parameters: {
        qubits: config.qubits,
        iterations: config.iterations
      },
      input_data: []
    });
  }

  /**
   * Get quantum simulation results
   */
  getQuantumResults(simulationId: string): Observable<any> {
    return this.get(`/quantum/results/${simulationId}`);
  }

  /**
   * Mathematical processing - Riemann Zeta
   */
  processZetaFunction(parameters: any): Observable<any> {
    console.log('📐 Processing Zeta function:', parameters);
    return this.post('/zeta-prediction', {
      algorithm: 'zeta_prediction',
      parameters: parameters,
      input_data: []
    });
  }

  /**
   * Get performance analytics
   */
  getAnalytics(timeRange: string = '24h'): Observable<any> {
    const params = new HttpParams().set('range', timeRange);
    return this.get('/analytics', params);
  }

  /**
   * Plugin API methods
   */
  
  /**
   * Get plugin health
   */
  getPluginHealth(): Observable<any> {
    return this.get('/plugin/health');
  }

  /**
   * Get tool catalog
   */
  getToolCatalog(): Observable<any> {
    return this.get('/plugin/catalog');
  }

  /**
   * Execute plugin tool
   */
  executePluginTool(toolName: string, parameters: any): Observable<any> {
    console.log('🔧 Executing plugin tool:', toolName, parameters);
    return this.post('/plugin/execute', {
      tool_name: toolName,
      parameters: parameters
    });
  }

  /**
   * Batch execute plugin tools
   */
  batchExecuteTools(tools: Array<{ tool_name: string; parameters: any }>): Observable<any> {
    console.log('🔧 Batch executing tools:', tools);
    return this.post('/plugin/batch-execute', { tools });
  }

  // Utility methods

  /**
   * Upload file
   */
  uploadFile(file: File, endpoint: string = '/upload'): Observable<any> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    
    return this.post(endpoint, formData);
  }

  /**
   * Download file
   */
  downloadFile(url: string): Observable<Blob> {
    return this.http.get(url, { responseType: 'blob' });
  }

  // Private methods

  private setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }

  private clearError(): void {
    this.errorSubject.next(null);
  }

  private handleError(error: any): Observable<never> {
    this.setLoading(false);
    
    let errorMessage = 'An unknown error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = error.error.message;
    } else {
      // Server-side error
      errorMessage = error.error?.message || error.message || `Error ${error.status}: ${error.statusText}`;
    }

    console.error('🔥 API Error:', error);
    console.error('📝 Error message:', errorMessage);

    this.errorSubject.next(errorMessage);
    
    return throwError(() => new Error(errorMessage));
  }

  // Connection testing
  testConnection(): Observable<boolean> {
    return new Observable(observer => {
      this.healthCheck().subscribe({
        next: (response) => {
          console.log('✅ API connection successful:', response);
          observer.next(true);
          observer.complete();
        },
        error: (error) => {
          console.error('❌ API connection failed:', error);
          observer.next(false);
          observer.complete();
        }
      });
    });
  }

  // Mathematical utilities
  calculateGoldenRatio(): number {
    return (1 + Math.sqrt(5)) / 2; // φ = 1.618034...
  }

  calculateSilverRatio(): number {
    return 1 / this.calculateGoldenRatio() - 1; // σ = 0.381966...
  }

  // Performance monitoring
  measureApiPerformance<T>(apiCall: Observable<T>): Observable<{ data: T; responseTime: number }> {
    const startTime = performance.now();
    
    return apiCall.pipe(
      tap((data) => {
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        console.log(`⏱️ API Response Time: ${responseTime.toFixed(2)}ms`);
      })
    ).pipe(
      tap(data => ({ data, responseTime: performance.now() - startTime }))
    ) as Observable<{ data: T; responseTime: number }>;
  }
}
