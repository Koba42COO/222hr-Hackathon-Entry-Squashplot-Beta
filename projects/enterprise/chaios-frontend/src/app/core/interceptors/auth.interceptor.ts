import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  
  // Get the auth token from localStorage
  const token = localStorage.getItem('access_token');
  
  if (token && authService.isAuthenticated()) {
    // Clone the request and add the authorization header
    const authReq = req.clone({
      headers: req.headers.set('Authorization', `Bearer ${token}`)
    });
    
    console.log('🔐 Adding auth token to request:', req.url);
    return next(authReq);
  }
  
  return next(req);
};

