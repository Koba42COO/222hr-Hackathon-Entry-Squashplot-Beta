import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
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
  IonIcon
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { mailOutline, arrowBackOutline } from 'ionicons/icons';

@Component({
  selector: 'app-forgot-password',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>Forgot Password</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true">
      <div class="forgot-password-container">
        <ion-card>
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="mail-outline"></ion-icon>
              Reset Password
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-text>
              <p>Enter your email address and we'll send you a link to reset your password.</p>
            </ion-text>
            
            <ion-item>
              <ion-icon name="mail-outline" slot="start"></ion-icon>
              <ion-label position="stacked">Email</ion-label>
              <ion-input type="email" [(ngModel)]="email" required></ion-input>
            </ion-item>

            <ion-button expand="block" (click)="sendResetEmail()" class="reset-button">
              <ion-icon name="mail-outline" slot="start"></ion-icon>
              Send Reset Link
            </ion-button>

            <ion-button expand="block" fill="clear" routerLink="/auth/login">
              <ion-icon name="arrow-back-outline" slot="start"></ion-icon>
              Back to Login
            </ion-button>
          </ion-card-content>
        </ion-card>
      </div>
    </ion-content>
  `,
  styleUrls: ['./forgot-password.page.scss'],
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
export class ForgotPasswordPage {
  email = '';

  constructor() {
    this.initializeIcons();
  }

  private initializeIcons() {
    addIcons({
      'mail-outline': mailOutline,
      'arrow-back-outline': arrowBackOutline
    });
  }

  sendResetEmail() {
    console.log('Sending reset email to:', this.email);
    // TODO: Implement password reset functionality
  }
}
