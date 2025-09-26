import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
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
  IonText,
  IonIcon,
  ToastController
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { personAddOutline, mailOutline, personOutline, lockClosedOutline } from 'ionicons/icons';

import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-register',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>Register</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true">
      <div class="register-container">
        <ion-card>
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="person-add-outline"></ion-icon>
              Create Account
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <form (ngSubmit)="onRegister()" #registerForm="ngForm">
              <ion-item>
                <ion-icon name="person-outline" slot="start"></ion-icon>
                <ion-label position="stacked">Full Name</ion-label>
                <ion-input
                  type="text"
                  [(ngModel)]="registerData.name"
                  name="name"
                  required>
                </ion-input>
              </ion-item>

              <ion-item>
                <ion-icon name="person-outline" slot="start"></ion-icon>
                <ion-label position="stacked">Username</ion-label>
                <ion-input
                  type="text"
                  [(ngModel)]="registerData.username"
                  name="username"
                  required>
                </ion-input>
              </ion-item>

              <ion-item>
                <ion-icon name="mail-outline" slot="start"></ion-icon>
                <ion-label position="stacked">Email</ion-label>
                <ion-input
                  type="email"
                  [(ngModel)]="registerData.email"
                  name="email"
                  required>
                </ion-input>
              </ion-item>

              <ion-item>
                <ion-icon name="lock-closed-outline" slot="start"></ion-icon>
                <ion-label position="stacked">Password</ion-label>
                <ion-input
                  type="password"
                  [(ngModel)]="registerData.password"
                  name="password"
                  required>
                </ion-input>
              </ion-item>

              <ion-item>
                <ion-icon name="lock-closed-outline" slot="start"></ion-icon>
                <ion-label position="stacked">Confirm Password</ion-label>
                <ion-input
                  type="password"
                  [(ngModel)]="registerData.confirmPassword"
                  name="confirmPassword"
                  required>
                </ion-input>
              </ion-item>

              <ion-button 
                expand="block" 
                type="submit" 
                [disabled]="!registerForm.valid || isLoading"
                class="register-button">
                <ion-icon name="person-add-outline" slot="start"></ion-icon>
                {{ isLoading ? 'Creating Account...' : 'Register' }}
              </ion-button>
            </form>

            <ion-text class="login-link">
              <p>Already have an account? <a routerLink="/auth/login">Sign in</a></p>
            </ion-text>
          </ion-card-content>
        </ion-card>
      </div>
    </ion-content>
  `,
  styleUrls: ['./register.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
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
    IonText,
    IonIcon
  ]
})
export class RegisterPage implements OnInit {
  registerData: RegisterData = {
    name: '',
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  };

  isLoading = false;

  constructor(
    private authService: AuthService,
    private router: Router,
    private toastController: ToastController
  ) {
    this.initializeIcons();
  }

  ngOnInit() {
  }

  private initializeIcons() {
    addIcons({
      'person-add-outline': personAddOutline,
      'mail-outline': mailOutline,
      'person-outline': personOutline,
      'lock-closed-outline': lockClosedOutline
    });
  }

  async onRegister() {
    if (this.registerData.password !== this.registerData.confirmPassword) {
      const toast = await this.toastController.create({
        message: 'Passwords do not match',
        duration: 3000,
        color: 'danger'
      });
      await toast.present();
      return;
    }

    this.isLoading = true;

    this.authService.register(this.registerData).subscribe({
      next: async (response) => {
        if (response.user) {
          const toast = await this.toastController.create({
            message: 'Account created successfully!',
            duration: 3000,
            color: 'success'
          });
          await toast.present();
          this.router.navigate(['/dashboard']);
        }
      },
      error: async (error) => {
        const toast = await this.toastController.create({
          message: error.message || 'Registration failed',
          duration: 3000,
          color: 'danger'
        });
        await toast.present();
        this.isLoading = false;
      },
      complete: () => {
        this.isLoading = false;
      }
    });
  }
}
