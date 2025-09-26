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
  IonIcon
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { lockClosedOutline, checkmarkCircleOutline } from 'ionicons/icons';

@Component({
  selector: 'app-reset-password',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>Reset Password</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true">
      <div class="reset-password-container">
        <ion-card>
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="lock-closed-outline"></ion-icon>
              Set New Password
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-item>
              <ion-icon name="lock-closed-outline" slot="start"></ion-icon>
              <ion-label position="stacked">New Password</ion-label>
              <ion-input type="password" [(ngModel)]="newPassword" required></ion-input>
            </ion-item>

            <ion-item>
              <ion-icon name="lock-closed-outline" slot="start"></ion-icon>
              <ion-label position="stacked">Confirm Password</ion-label>
              <ion-input type="password" [(ngModel)]="confirmPassword" required></ion-input>
            </ion-item>

            <ion-button expand="block" (click)="resetPassword()" class="reset-button">
              <ion-icon name="checkmark-circle-outline" slot="start"></ion-icon>
              Reset Password
            </ion-button>
          </ion-card-content>
        </ion-card>
      </div>
    </ion-content>
  `,
  styleUrls: ['./reset-password.page.scss'],
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
    IonIcon
  ]
})
export class ResetPasswordPage {
  newPassword = '';
  confirmPassword = '';

  constructor() {
    this.initializeIcons();
  }

  private initializeIcons() {
    addIcons({
      'lock-closed-outline': lockClosedOutline,
      'checkmark-circle-outline': checkmarkCircleOutline
    });
  }

  resetPassword() {
    if (this.newPassword !== this.confirmPassword) {
      console.log('Passwords do not match');
      return;
    }
    console.log('Resetting password...');
    // TODO: Implement password reset functionality
  }
}
