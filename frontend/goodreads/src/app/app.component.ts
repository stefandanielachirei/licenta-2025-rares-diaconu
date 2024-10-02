import { Component } from '@angular/core';
import { DataService } from './data.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'goodreads';
  flaskText: string = '';

  constructor(private dataService: DataService) {}

  getText(){
    this.dataService.getTextFromFlask().subscribe(
      (response) => {
        if(response.text){
          this.flaskText = response.text;
        }
        else{
          this.flaskText = 'No text available';
        }
      },
      (error) => {
        console.error('Error fetching text from Flask API', error);
        this.flaskText = 'Error fetching text!';
      }
    );
  }
}
