import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../../core/services/auth.service';
import { AlertController } from '@ionic/angular';

@Component({
  selector: 'app-register',
  templateUrl: './register.page.html',
  styleUrls: ['./register.page.scss'],
})
export class RegisterPage implements OnInit {
  registerForm: FormGroup;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router,
    private alertController: AlertController
  ) {
    this.registerForm = this.fb.group({
      username: ['', [Validators.required]],
      email: ['', [Validators.required, Validators.email]],
      fullName: ['', [Validators.required]],
      password: ['', [Validators.required, Validators.minLength(6)]],
    });
  }

  ngOnInit() {
  }

  async onRegister() {
    if (this.registerForm.invalid) {
      return;
    }

    const { username, email, fullName, password } = this.registerForm.value;

    this.authService.register(username, email, fullName, password).subscribe({
      next: (response) => {
        this.router.navigate(['/llm-convo']);
      },
      error: async (error) => {
        const alert = await this.alertController.create({
          header: 'Registration Failed',
          message: error.error.detail || 'An unknown error occurred.',
          buttons: ['OK'],
        });
        await alert.present();
      }
    });
  }
}
