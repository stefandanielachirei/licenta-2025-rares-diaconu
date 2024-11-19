import { Component } from '@angular/core';
import { DataService } from './data.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'goodreads';
  fastAPIText: string = '';

  constructor(private dataService: DataService) {}

  getText(){
    this.dataService.getTextFromFastAPI().subscribe(
      (response) => {
        if(response.text){
          this.fastAPIText = response.text;
        }
        else{
          this.fastAPIText = 'No text available';
        }
      },
      (error) => {
        console.error('Error fetching text from Flask API', error);
        this.fastAPIText = 'Error fetching text!';
      }
    );
  }
}
