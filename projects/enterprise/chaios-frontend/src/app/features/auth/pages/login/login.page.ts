import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { 
  IonContent, 
  IonHeader, 
  IonTitle, 
  IonToolbar, 
  IonCard, 
  IonCardHeader, 
  IonCardTitle, 
  IonCardContent, 
  IonItem, 
  IonLabel, 
  IonInput, 
  IonButton, 
  IonCheckbox, 
  IonText,
  IonIcon,
  IonGrid,
  IonRow,
  IonCol,
  IonSpinner
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  mailOutline, 
  lockClosedOutline, 
  eyeOutline, 
  eyeOffOutline,
  sparklesOutline,
  logoGoogle,
  logoApple,
  logoGithub
} from 'ionicons/icons';

import { AuthService } from '../../core/auth.service';
import { NotificationService } from '../../../../core/services/notification.service';

@Component({
  selector: 'app-login',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar color="primary">
        <ion-title>chAIos Login</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true" class="login-content">
      <div class="login-container">
        
        <!-- Logo and Welcome Section -->
        <div class="welcome-section">
          <div class="logo-container">
            <ion-icon name="sparkles-outline" class="logo-icon"></ion-icon>
            <h1 class="platform-title">chAIos</h1>
            <p class="platform-subtitle">Chiral Harmonic Aligned Intelligence</p>
          </div>
        </div>

        <ion-grid>
          <ion-row class="ion-justify-content-center">
            <ion-col size="12" size-md="6" size-lg="4">
              
              <!-- Login Form Card -->
              <ion-card class="login-card">
                <ion-card-header>
                  <ion-card-title class="card-title">
                    <ion-icon name="lock-closed-outline"></ion-icon>
                    Welcome Back
                  </ion-card-title>
                </ion-card-header>

                <ion-card-content>
                  <form [formGroup]="loginForm" (ngSubmit)="onSubmit()">
                    
                    <!-- Email Field -->
                    <ion-item class="form-item">
                      <ion-icon name="mail-outline" slot="start" class="field-icon"></ion-icon>
                      <ion-label position="stacked">Email</ion-label>
                      <ion-input 
                        type="email" 
                        formControlName="email"
                        placeholder="Enter your email"
                        class="form-input">
                      </ion-input>
                    </ion-item>

                    <!-- Password Field -->
                    <ion-item class="form-item">
                      <ion-icon name="lock-closed-outline" slot="start" class="field-icon"></ion-icon>
                      <ion-label position="stacked">Password</ion-label>
                      <ion-input 
                        [type]="showPassword ? 'text' : 'password'" 
                        formControlName="password"
                        placeholder="Enter your password"
                        class="form-input">
                      </ion-input>
                      <ion-button 
                        fill="clear" 
                        slot="end"
                        (click)="togglePasswordVisibility()"
                        class="password-toggle">
                        <ion-icon [name]="showPassword ? 'eye-off-outline' : 'eye-outline'"></ion-icon>
                      </ion-button>
                    </ion-item>

                    <!-- Remember Me -->
                    <ion-item class="checkbox-item">
                      <ion-checkbox 
                        formControlName="rememberMe"
                        slot="start">
                      </ion-checkbox>
                      <ion-label class="checkbox-label">Remember me</ion-label>
                    </ion-item>

                    <!-- Login Button -->
                    <ion-button 
                      expand="block" 
                      type="submit" 
                      [disabled]="!loginForm.valid || isLoading"
                      class="login-button">
                      <ion-spinner *ngIf="isLoading" name="crescent"></ion-spinner>
                      <span *ngIf="!isLoading">Sign In</span>
                    </ion-button>

                    <!-- Forgot Password -->
                    <div class="forgot-password">
                      <ion-button 
                        fill="clear" 
                        size="small"
                        [routerLink]="['/auth/forgot-password']">
                        Forgot password?
                      </ion-button>
                    </div>

                  </form>
                </ion-card-content>
              </ion-card>

              <!-- Register Link -->
              <div class="register-section">
                <ion-text class="register-text">
                  <p>Don't have an account? 
                    <ion-button 
                      fill="clear" 
                      size="small"
                      [routerLink]="['/auth/register']"
                      class="register-link">
                      Create one
                    </ion-button>
                  </p>
                </ion-text>
              </div>

            </ion-col>
          </ion-row>
        </ion-grid>

      </div>
    </ion-content>
  `,
  styleUrls: ['./login.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    IonContent,
    IonHeader,
    IonTitle,
    IonToolbar,
    IonCard,
    IonCardHeader,
    IonCardTitle,
    IonCardContent,
    IonItem,
    IonLabel,
    IonInput,
    IonButton,
    IonCheckbox,
    IonText,
    IonIcon,
    IonGrid,
    IonRow,
    IonCol,
    IonSpinner
  ]
})
export class LoginPage implements OnInit {
  loginForm: FormGroup;
  showPassword = false;
  isLoading = false;

  constructor(
    private formBuilder: FormBuilder,
    private authService: AuthService,
    private notificationService: NotificationService,
    private router: Router
  ) {
    this.initializeIcons();
    this.loginForm = this.createLoginForm();
  }

  ngOnInit() {
    // Check if user is already authenticated
    if (this.authService.isAuthenticated()) {
      this.router.navigate(['/dashboard']);
    }
  }

  private initializeIcons() {
    addIcons({
      'mail-outline': mailOutline,
      'lock-closed-outline': lockClosedOutline,
      'eye-outline': eyeOutline,
      'eye-off-outline': eyeOffOutline,
      'sparkles-outline': sparklesOutline,
      'logo-google': logoGoogle,
      'logo-apple': logoApple,
      'logo-github': logoGithub
    });
  }

  private createLoginForm(): FormGroup {
    return this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      rememberMe: [false]
    });
  }

  togglePasswordVisibility() {
    this.showPassword = !this.showPassword;
  }

  async onSubmit() {
    if (this.loginForm.valid && !this.isLoading) {
      this.isLoading = true;
      
      const credentials: LoginCredentials = {
        email: this.loginForm.get('email')?.value,
        password: this.loginForm.get('password')?.value,
        rememberMe: this.loginForm.get('rememberMe')?.value
      };

      try {
        await this.authService.login(credentials).toPromise();
        
        this.notificationService.showNotification(
          'Welcome to chAIos! Consciousness matrix activated.', 
          'success'
        );
        
        // Navigate to dashboard
        this.router.navigate(['/dashboard']);
        
      } catch (error: any) {
        console.error('Login error:', error);
        
        const errorMessage = error.message || 'Login failed. Please check your credentials.';
        this.notificationService.showNotification(errorMessage, 'error');
        
      } finally {
        this.isLoading = false;
      }
    } else {
      this.markFormGroupTouched();
    }
  }

  async loginDemo() {
    this.isLoading = true;
    
    try {
      // Use the demo login from AuthService
      this.authService.loginAsDemo();
      
      this.notificationService.showNotification(
        'Demo mode activated! Exploring consciousness mathematics...', 
        'success'
      );
      
      // Navigate to dashboard
      this.router.navigate(['/dashboard']);
      
    } catch (error) {
      console.error('Demo login error:', error);
      this.notificationService.showNotification('Demo login failed.', 'error');
    } finally {
      this.isLoading = false;
    }
  }

  // Social login methods (placeholder for future implementation)
  async loginWithGoogle() {
    this.notificationService.showNotification('Google login coming soon!', 'info');
  }

  async loginWithApple() {
    this.notificationService.showNotification('Apple login coming soon!', 'info');
  }

  async loginWithGithub() {
    this.notificationService.showNotification('GitHub login coming soon!', 'info');
  }

  private markFormGroupTouched() {
    Object.keys(this.loginForm.controls).forEach(key => {
      const control = this.loginForm.get(key);
      control?.markAsTouched();
    });
  }

  // Form validation helpers
  isFieldInvalid(fieldName: string): boolean {
    const field = this.loginForm.get(fieldName);
    return !!(field && field.invalid && field.touched);
  }

  getFieldError(fieldName: string): string {
    const field = this.loginForm.get(fieldName);
    
    if (field?.errors) {
      if (field.errors['required']) {
        return `${this.getFieldDisplayName(fieldName)} is required`;
      }
      if (field.errors['email']) {
        return 'Please enter a valid email address';
      }
      if (field.errors['minlength']) {
        return `Password must be at least ${field.errors['minlength'].requiredLength} characters`;
      }
    }
    
    return '';
  }

  private getFieldDisplayName(fieldName: string): string {
    const displayNames: { [key: string]: string } = {
      email: 'Email',
      password: 'Password'
    };
    return displayNames[fieldName] || fieldName;
  }
}

