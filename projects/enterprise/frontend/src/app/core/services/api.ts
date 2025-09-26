import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry, map } from 'rxjs/operators';
import { environment } from '../../../environments/environment';

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ProcessingRequest {
  input_type: string;
  algorithm: string;
  parameters: Record<string, any>;
  data: any;
}

export interface ProcessingResponse {
  success: boolean;
  result?: any;
  processing_time: number;
  timestamp: string;
  error?: string;
}

export interface HealthResponse {
  status: string;
  uptime: number;
  timestamp: string;
  version: string;
  services: Record<string, boolean>;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly baseUrl = environment.apiUrl || 'http://localhost:8000';
  
  constructor(private http: HttpClient) {}

  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('auth_token');
    let headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    
    if (token) {
      headers = headers.set('Authorization', `Bearer ${token}`);
    }
    
    return headers;
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An unknown error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = error.error.message;
    } else {
      // Server-side error
      errorMessage = error.error?.error || error.message || `Error Code: ${error.status}`;
    }
    
    console.error('API Error:', errorMessage);
    return throwError(() => errorMessage);
  }

  // Authentication endpoints
  login(email: string, password: string): Observable<ApiResponse> {
    return this.http.post<any>(`${this.baseUrl}/auth/login`, {
      username: email,  // Backend expects username field, but we're using email as username
      password
    }).pipe(
      retry(1),
      map((response) => {
        // Transform the backend response to match frontend expectations
        if (response.access_token) {
          return {
            success: true,
            data: {
              token: response.access_token,
              user: response.user
            }
          };
        }
        return response;
      }),
      catchError(this.handleError)
    );
  }

  register(email: string, password: string, name: string): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/auth/register`, {
      email,
      password,
      name
    }).pipe(
      retry(1),
      catchError(this.handleError)
    );
  }

  // Consciousness processing endpoints
  processConsciousness(request: ProcessingRequest): Observable<ProcessingResponse> {
    return this.http.post<ProcessingResponse>(`${this.baseUrl}/consciousness/process`, request, {
      headers: this.getHeaders()
    }).pipe(
      retry(1),
      catchError(this.handleError)
    );
  }

  // System health
  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`).pipe(
      catchError(this.handleError)
    );
  }

  // System metrics
  getMetrics(): Observable<any> {
    return this.http.get(`${this.baseUrl}/metrics`, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  // Chat message processing
  sendChatMessage(message: string): Observable<ProcessingResponse> {
    const request: ProcessingRequest = {
      input_type: 'text',
      algorithm: 'wallace_transform',
      parameters: {
        consciousness_level: 0.95,
        response_style: 'conversational'
      },
      data: message
    };

    return this.processConsciousness(request);
  }

  // Admin endpoints
  getAdminUsers(): Observable<ApiResponse> {
    return this.http.get<ApiResponse>(`${this.baseUrl}/admin/users`, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  createUserAdmin(userData: any): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/admin/users`, userData, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  deleteUserAdmin(userId: string): Observable<ApiResponse> {
    return this.http.delete<ApiResponse>(`${this.baseUrl}/admin/users/${userId}`, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  getAdminSystemStats(): Observable<ApiResponse> {
    return this.http.get<ApiResponse>(`${this.baseUrl}/admin/system-stats`, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  createDefaultAdmin(): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/admin/create-default-admin`, {}).pipe(
      catchError(this.handleError)
    );
  }

  // System Tools Integration
  getSystemTools(): Observable<ApiResponse> {
    return this.http.get<ApiResponse>(`${this.baseUrl}/system-tools/list`, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  executeSystemTool(toolName: string, parameters: any = {}, permissions: string[] = ['read', 'write', 'consciousness', 'system', 'development', 'ai_ml', 'security', 'integration', 'quantum', 'blockchain', 'grok_jr', 'admin']): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/system-tools/execute`, {
      tool_name: toolName,
      parameters: parameters,
      permissions: permissions
    }, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  batchExecuteSystemTools(toolSequence: any[], permissions: string[] = ['read', 'write', 'consciousness', 'system', 'development', 'ai_ml', 'security', 'integration', 'quantum', 'blockchain', 'grok_jr', 'admin']): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/system-tools/batch-execute`, {
      tool_sequence: toolSequence,
      permissions: permissions
    }, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }

  // Enhanced consciousness processing with system tools
  processConsciousnessWithTools(algorithm: string, parameters: any = {}): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.baseUrl}/consciousness/process`, {
      algorithm: algorithm,
      parameters: parameters
    }, {
      headers: this.getHeaders()
    }).pipe(
      catchError(this.handleError)
    );
  }
}
