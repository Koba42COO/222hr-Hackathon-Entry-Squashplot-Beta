import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, throwError, of, timer, EMPTY } from 'rxjs';
import { tap, catchError, map, switchMap, retry, timeout, distinctUntilChanged, filter, shareReplay } from 'rxjs/operators';
import { Router } from '@angular/router';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

import { environment } from '../../../environments/environment';
import { OrchestrationService } from './orchestration.service';

/**
 * chAIos Authentication Orchestrator Service
 * ==========================================
 * Following tangtalk repository patterns for authentication management
 * Implements proper separation of concerns and orchestration architecture
 */

// Authentication Interfaces
export interface User {
  id: string;
  email: string;
  name: string;
  username: string;
  role: 'user' | 'admin' | 'moderator';
  permissions: string[];
  avatar?: string;
  lastLogin?: Date;
  preferences: {
    theme: 'light' | 'dark' | 'consciousness';
    notifications: boolean;
    language: string;
  };
  consciousness: {
    level: number;
    experience: number;
    achievements: string[];
  };
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  email: string;
  password: string;
  confirmPassword: string;
  name: string;
  username: string;
  acceptTerms: boolean;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: string;
}

export interface AuthResponse {
  tokens: AuthTokens;
  user: User;
  message: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  tokens: AuthTokens | null;
  loading: boolean;
  error: string | null;
  lastActivity: Date | null;
  sessionExpiry: Date | null;
}

@Injectable({
  providedIn: 'root'
})
export class AuthOrchestratorService {
  // Private State Management
  // ========================
  private readonly STORAGE_KEYS = {
    TOKENS: 'chaios_auth_tokens',
    USER: 'chaios_user_data',
    REMEMBER_ME: 'chaios_remember_me'
  } as const;

  private authState$ = new BehaviorSubject<AuthState>({
    isAuthenticated: false,
    user: null,
    tokens: null,
    loading: false,
    error: null,
    lastActivity: null,
    sessionExpiry: null
  });

  private tokenRefreshTimer: any;
  private sessionTimeoutTimer: any;
  private activityCheckInterval: any;

  // Public Observables
  // ==================
  public readonly currentUser$: Observable<User | null> = this.authState$.pipe(
    map(state => state.user),
    distinctUntilChanged()
  );

  public readonly isAuthenticated$: Observable<boolean> = this.authState$.pipe(
    map(state => state.isAuthenticated),
    distinctUntilChanged()
  );

  public readonly isLoading$: Observable<boolean> = this.authState$.pipe(
    map(state => state.loading),
    distinctUntilChanged()
  );

  public readonly authError$: Observable<string | null> = this.authState$.pipe(
    map(state => state.error),
    distinctUntilChanged()
  );

  public readonly sessionExpiry$: Observable<Date | null> = this.authState$.pipe(
    map(state => state.sessionExpiry),
    distinctUntilChanged()
  );

  constructor(
    private http: HttpClient,
    private router: Router,
    private orchestration: OrchestrationService
  ) {
    this.initializeAuthOrchestrator();
  }

  /**
   * Initialize Authentication Orchestrator
   * =====================================
   * Sets up authentication state, token management, and session monitoring
   */
  private initializeAuthOrchestrator(): void {
    console.log('🔐 Initializing chAIos Authentication Orchestrator');

    // Check for existing authentication
    this.checkExistingAuth();

    // Set up automatic token refresh
    this.setupTokenRefresh();

    // Set up session monitoring
    this.setupSessionMonitoring();

    // Set up activity tracking
    this.setupActivityTracking();

    console.log('✅ Authentication Orchestrator initialized');
  }

  /**
   * Authentication Methods
   * =====================
   */

  /**
   * Login with credentials
   */
  login(credentials: LoginCredentials): Observable<AuthResponse> {
    this.updateAuthState({ loading: true, error: null });

    return this.http.post<AuthResponse>(`${environment.apiUrl}/auth/login`, credentials).pipe(
      timeout(10000),
      retry(2),
      tap((response) => {
        this.handleAuthSuccess(response, credentials.rememberMe);
        console.log('✅ Login successful');
      }),
      catchError((error: HttpErrorResponse) => {
        this.handleAuthError('Login failed', error);
        return throwError(() => error);
      }),
      shareReplay(1)
    );
  }

  /**
   * Register new user
   */
  register(registerData: RegisterData): Observable<AuthResponse> {
    this.updateAuthState({ loading: true, error: null });

    return this.http.post<AuthResponse>(`${environment.apiUrl}/auth/register`, registerData).pipe(
      timeout(10000),
      retry(2),
      tap((response) => {
        this.handleAuthSuccess(response, false);
        console.log('✅ Registration successful');
      }),
      catchError((error: HttpErrorResponse) => {
        this.handleAuthError('Registration failed', error);
        return throwError(() => error);
      }),
      shareReplay(1)
    );
  }

  /**
   * Logout user
   */
  logout(): Observable<void> {
    const currentState = this.authState$.value;
    
    if (!currentState.isAuthenticated) {
      return of(void 0);
    }

    this.updateAuthState({ loading: true });

    return this.http.post<void>(`${environment.apiUrl}/auth/logout`, {
      refreshToken: currentState.tokens?.refreshToken
    }).pipe(
      timeout(5000),
      tap(() => {
        this.handleLogout();
        console.log('✅ Logout successful');
      }),
      catchError((error) => {
        // Even if logout fails on server, clear local state
        this.handleLogout();
        console.warn('⚠️ Logout completed locally due to server error:', error);
        return of(void 0);
      })
    );
  }

  /**
   * Refresh authentication token
   */
  refreshToken(): Observable<AuthTokens> {
    const currentState = this.authState$.value;
    const refreshToken = currentState.tokens?.refreshToken;

    if (!refreshToken) {
      return throwError(() => new Error('No refresh token available'));
    }

    return this.http.post<{ tokens: AuthTokens }>(`${environment.apiUrl}/auth/refresh`, {
      refreshToken
    }).pipe(
      timeout(10000),
      map(response => response.tokens),
      tap((tokens) => {
        this.updateTokens(tokens);
        this.setupTokenRefresh();
        console.log('✅ Token refreshed successfully');
      }),
      catchError((error) => {
        console.error('❌ Token refresh failed:', error);
        this.handleLogout();
        return throwError(() => error);
      })
    );
  }

  /**
   * Check if user has specific permission
   */
  hasPermission(permission: string): boolean {
    const user = this.authState$.value.user;
    if (!user) return false;
    
    return user.permissions.includes(permission) || 
           user.permissions.includes('admin') ||
           user.role === 'admin';
  }

  /**
   * Check if user has specific role
   */
  hasRole(role: string): boolean {
    const user = this.authState$.value.user;
    if (!user) return false;
    
    return user.role === role || user.role === 'admin';
  }

  /**
   * Get current user data
   */
  getCurrentUser(): Observable<User | null> {
    return this.currentUser$;
  }

  /**
   * Update user profile
   */
  updateProfile(profileData: Partial<User>): Observable<User> {
    const currentState = this.authState$.value;
    
    if (!currentState.isAuthenticated || !currentState.user) {
      return throwError(() => new Error('User not authenticated'));
    }

    return this.http.put<User>(`${environment.apiUrl}/auth/profile`, profileData).pipe(
      timeout(10000),
      tap((updatedUser) => {
        this.updateAuthState({
          user: { ...currentState.user!, ...updatedUser }
        });
        this.saveUserToStorage(updatedUser);
        console.log('✅ Profile updated successfully');
      }),
      catchError((error) => {
        this.handleAuthError('Profile update failed', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Change password
   */
  changePassword(currentPassword: string, newPassword: string): Observable<void> {
    return this.http.post<void>(`${environment.apiUrl}/auth/change-password`, {
      currentPassword,
      newPassword
    }).pipe(
      timeout(10000),
      tap(() => {
        console.log('✅ Password changed successfully');
      }),
      catchError((error) => {
        this.handleAuthError('Password change failed', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Private Helper Methods
   * =====================
   */

  private checkExistingAuth(): void {
    const savedTokens = this.getTokensFromStorage();
    const savedUser = this.getUserFromStorage();

    if (savedTokens && savedUser && this.isTokenValid(savedTokens)) {
      this.updateAuthState({
        isAuthenticated: true,
        user: savedUser,
        tokens: savedTokens,
        sessionExpiry: new Date(Date.now() + savedTokens.expiresIn * 1000),
        lastActivity: new Date()
      });

      this.setupTokenRefresh();
      console.log('✅ Existing authentication restored');
    } else {
      this.clearStoredAuth();
    }
  }

  private handleAuthSuccess(response: AuthResponse, rememberMe: boolean = false): void {
    const sessionExpiry = new Date(Date.now() + response.tokens.expiresIn * 1000);

    this.updateAuthState({
      isAuthenticated: true,
      user: response.user,
      tokens: response.tokens,
      loading: false,
      error: null,
      sessionExpiry,
      lastActivity: new Date()
    });

    // Save to storage if remember me is enabled
    if (rememberMe) {
      this.saveAuthToStorage(response.tokens, response.user);
    }

    // Set up automatic token refresh
    this.setupTokenRefresh();

    // Notify orchestration service
    this.orchestration.registerService('auth', this);
  }

  private handleAuthError(message: string, error: HttpErrorResponse): void {
    let errorMessage = message;

    if (error.error?.message) {
      errorMessage = error.error.message;
    } else if (error.status === 401) {
      errorMessage = 'Invalid credentials';
    } else if (error.status === 403) {
      errorMessage = 'Access forbidden';
    } else if (error.status === 429) {
      errorMessage = 'Too many attempts. Please try again later.';
    }

    this.updateAuthState({
      loading: false,
      error: errorMessage
    });

    console.error('❌ Authentication error:', error);
  }

  private handleLogout(): void {
    this.clearTokenRefreshTimer();
    this.clearSessionTimeout();
    this.clearActivityTracking();
    this.clearStoredAuth();

    this.updateAuthState({
      isAuthenticated: false,
      user: null,
      tokens: null,
      loading: false,
      error: null,
      lastActivity: null,
      sessionExpiry: null
    });

    this.router.navigate(['/auth/login']);
  }

  private updateAuthState(partialState: Partial<AuthState>): void {
    const currentState = this.authState$.value;
    this.authState$.next({ ...currentState, ...partialState });
  }

  private updateTokens(tokens: AuthTokens): void {
    const currentState = this.authState$.value;
    const sessionExpiry = new Date(Date.now() + tokens.expiresIn * 1000);

    this.updateAuthState({
      tokens,
      sessionExpiry
    });

    if (this.getRememberMeFromStorage()) {
      this.saveTokensToStorage(tokens);
    }
  }

  private setupTokenRefresh(): void {
    this.clearTokenRefreshTimer();

    const tokens = this.authState$.value.tokens;
    if (!tokens) return;

    // Refresh token 5 minutes before expiry
    const refreshTime = (tokens.expiresIn - 300) * 1000;
    
    this.tokenRefreshTimer = setTimeout(() => {
      this.refreshToken().subscribe({
        error: (error) => {
          console.error('❌ Automatic token refresh failed:', error);
          this.handleLogout();
        }
      });
    }, refreshTime);
  }

  private setupSessionMonitoring(): void {
    // Monitor for session expiry
    this.sessionExpiry$.pipe(
      filter(expiry => expiry !== null),
      switchMap(expiry => timer(expiry!.getTime() - Date.now()))
    ).subscribe(() => {
      console.warn('⚠️ Session expired');
      this.handleLogout();
    });
  }

  private setupActivityTracking(): void {
    // Track user activity for session management
    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
    
    const updateActivity = () => {
      if (this.authState$.value.isAuthenticated) {
        this.updateAuthState({ lastActivity: new Date() });
      }
    };

    activityEvents.forEach(event => {
      document.addEventListener(event, updateActivity, true);
    });

    // Check for inactivity every minute
    this.activityCheckInterval = setInterval(() => {
      const lastActivity = this.authState$.value.lastActivity;
      if (lastActivity && Date.now() - lastActivity.getTime() > 30 * 60 * 1000) { // 30 minutes
        console.warn('⚠️ User inactive - logging out');
        this.logout().subscribe();
      }
    }, 60000);
  }

  private isTokenValid(tokens: AuthTokens): boolean {
    const now = Date.now();
    const tokenExpiry = now + (tokens.expiresIn * 1000);
    return tokenExpiry > now + (5 * 60 * 1000); // Valid if more than 5 minutes remaining
  }

  private saveAuthToStorage(tokens: AuthTokens, user: User): void {
    localStorage.setItem(this.STORAGE_KEYS.TOKENS, JSON.stringify(tokens));
    localStorage.setItem(this.STORAGE_KEYS.USER, JSON.stringify(user));
    localStorage.setItem(this.STORAGE_KEYS.REMEMBER_ME, 'true');
  }

  private saveTokensToStorage(tokens: AuthTokens): void {
    localStorage.setItem(this.STORAGE_KEYS.TOKENS, JSON.stringify(tokens));
  }

  private saveUserToStorage(user: User): void {
    localStorage.setItem(this.STORAGE_KEYS.USER, JSON.stringify(user));
  }

  private getTokensFromStorage(): AuthTokens | null {
    try {
      const tokens = localStorage.getItem(this.STORAGE_KEYS.TOKENS);
      return tokens ? JSON.parse(tokens) : null;
    } catch {
      return null;
    }
  }

  private getUserFromStorage(): User | null {
    try {
      const user = localStorage.getItem(this.STORAGE_KEYS.USER);
      return user ? JSON.parse(user) : null;
    } catch {
      return null;
    }
  }

  private getRememberMeFromStorage(): boolean {
    return localStorage.getItem(this.STORAGE_KEYS.REMEMBER_ME) === 'true';
  }

  private clearStoredAuth(): void {
    localStorage.removeItem(this.STORAGE_KEYS.TOKENS);
    localStorage.removeItem(this.STORAGE_KEYS.USER);
    localStorage.removeItem(this.STORAGE_KEYS.REMEMBER_ME);
  }

  private clearTokenRefreshTimer(): void {
    if (this.tokenRefreshTimer) {
      clearTimeout(this.tokenRefreshTimer);
      this.tokenRefreshTimer = null;
    }
  }

  private clearSessionTimeout(): void {
    if (this.sessionTimeoutTimer) {
      clearTimeout(this.sessionTimeoutTimer);
      this.sessionTimeoutTimer = null;
    }
  }

  private clearActivityTracking(): void {
    if (this.activityCheckInterval) {
      clearInterval(this.activityCheckInterval);
      this.activityCheckInterval = null;
    }
  }

  /**
   * Cleanup
   * =======
   */
  ngOnDestroy(): void {
    this.clearTokenRefreshTimer();
    this.clearSessionTimeout();
    this.clearActivityTracking();
    this.authState$.complete();
    console.log('🧹 Authentication Orchestrator destroyed');
  }
}
