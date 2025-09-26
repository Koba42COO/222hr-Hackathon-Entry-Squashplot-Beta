import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RegisterPage } from './register.page';
import { RegisterPageRoutingModule } from './register-routing.module';
import { ReactiveFormsModule } from '@angular/forms';

@NgModule({
  imports: [
    CommonModule,
    RegisterPageRoutingModule,
    ReactiveFormsModule,
    RegisterPage
  ],
  declarations: []
})
export class RegisterPageModule {}
