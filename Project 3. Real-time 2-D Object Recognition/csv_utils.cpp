/*
  Xujia Qin 12th Feb, 2025
  S21
*/
#include <cstdio>
#include <cstring>
#include <vector>

/*
  Reads a string from a CSV file and stores it in the provided char array `os`.

  The function returns `false` (0) upon successfully reading a value.  
  It returns `true` (1) when reaching the end of a line or the file.
*/
int getstring( FILE *fp, char os[] ) {
  int p = 0;
  int eol = 0;

  for(;;) {
	char ch = fgetc( fp );
	if( ch == ',' ) {
	  break;
	}
	else if( ch == '\n' || ch == EOF ) {
	  eol = 1;
	  break;
	}
	// printf("%c", ch ); // uncomment for debugging
	os[p] = ch;
	p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return(eol); // return true if eol
}

/*
  Extracts an integer value from a CSV file and stores it in the provided pointer `v`.

  Returns `true` (1) if the end of a line or file is reached, otherwise `false` (0).
*/
int getint(FILE *fp, int *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
	char ch = fgetc( fp );
	if( ch == ',') {
	  break;
	}
	else if(ch == '\n' || ch == EOF) {
	  eol = 1;
	  break;
	}

	s[p] = ch;
	p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return(eol); // return true if eol
}

/*
  Reads a floating-point value from a CSV file and stores it in the provided pointer `v`.

  Returns `true` (1) when reaching the end of a line or the file, otherwise `false` (0).
*/
int getfloat(FILE *fp, double *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
	char ch = fgetc( fp );
	if( ch == ',') {
	  break;
	}
	else if(ch == '\n' || ch == EOF) {
	  eol = 1;
	  break;
	}

	s[p] = ch;
	p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return(eol); // return true if eol
}

/*
  Appends image feature data to a CSV file.  

  - `filename`: Name of the CSV file.  
  - `image_filename`: Name of the image file to be written in the first column.  
  - `image_data`: A vector of floating-point values representing image features.  
  - `reset_file`: If `true`, clears the file before writing; otherwise, appends to it.  

  Returns a nonzero value in case of an error.
*/
int append_image_data_csv( char *filename, char *image_filename, std::vector<double> &image_data, int reset_file ) {
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if( reset_file ) {
	strcpy( mode, "w" );
  }

  fp = fopen( filename, mode );
  if(!fp) {
	printf("Unable to open output file %s\n", filename );
	exit(-1);
  }

  // write the filename and the feature vector to the CSV file
  strcpy(buffer, image_filename);
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp );
  for(int i=0;i<image_data.size();i++) {
	char tmp[256];
	sprintf(tmp, ",%.4f", image_data[i] );
	std::fwrite(tmp, sizeof(char), strlen(tmp), fp );
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);

  return(0);
}

/*
  Reads image feature data from a CSV file into memory.  

  - `filename`: Name of the CSV file to read from.  
  - `filenames`: Vector to store image filenames.  
  - `data`: 2D vector to store feature values from the file.  
  - `echo_file`: If `true`, prints the data read from the file.  

  Returns a nonzero value in case of an error.
*/
int read_image_data_csv( char *filename, std::vector<char *> &filenames, std::vector<std::vector<double>> &data, int echo_file ) {
  FILE *fp;
  double fval;
  char img_file[256];

  fp = fopen(filename, "r");
  if( !fp ) {
	printf("Unable to open feature file\n");
	return(-1);
  }

  printf("Reading %s\n", filename);
  for(;;) {
	std::vector<double> dvec;


	// read the filename
	if( getstring( fp, img_file ) ) {
	  break;
	}
	// printf("Evaluting %s\n", filename);

	// read the whole feature file into memory
	for(;;) {
	  // get next feature
	  float eol = getfloat( fp, &fval );
	  dvec.push_back( fval );
	  if( eol ) break;
	}
	// printf("read %lu features\n", dvec.size() );

	data.push_back(dvec);

	char *fname = new char[strlen(img_file)+1];
	strcpy(fname, img_file);
	filenames.push_back( fname );
  }
  fclose(fp);
  printf("Finished reading CSV file\n");

  if(echo_file) {
	for(int i=0;i<data.size();i++) {
	  for(int j=0;j<data[i].size();j++) {
		printf("%.4f  ", data[i][j] );
	  }
	  printf("\n");
	}
	printf("\n");
  }

  return(0);
}