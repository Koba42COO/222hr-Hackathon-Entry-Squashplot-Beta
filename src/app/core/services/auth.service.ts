import { Injectable } from '@angular/core';
import { ApiService } from './api.service';
import { BehaviorSubject, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private authState = new BehaviorSubject<boolean>(this.hasToken());

  constructor(
    private apiService: ApiService,
    private router: Router
  ) { }

  isAuthenticated(): Observable<boolean> {
    return this.authState.asObservable();
  }

  private hasToken(): boolean {
    return !!localStorage.getItem('authToken');
  }

  login(username: string, password: string): Observable<any> {
    return this.apiService.post('auth/login', { username, password }).pipe(
      tap((response: any) => {
        localStorage.setItem('authToken', response.access_token);
        localStorage.setItem('refreshToken', response.refresh_token);
        this.authState.next(true);
      })
    );
  }

  register(username: string, email: string, fullName: string, password: string): Observable<any> {
    return this.apiService.post('auth/register', { username, email, full_name: fullName, password }).pipe(
      tap((response: any) => {
        // Auto-login after registration
        localStorage.setItem('authToken', response.access_token);
        localStorage.setItem('refreshToken', response.refresh_token);
        this.authState.next(true);
      })
    );
  }

  logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('refreshToken');
    this.authState.next(false);
    this.router.navigate(['/auth/login']);
  }
}
