#include <stdio.h>
#include "stm32f769i_discovery_lcd.h" 


/* Redirect the printf to the LCD */
#ifdef __GNUC__
/* With GCC/RAISONANCE, small printf (option LD Linker->Libraries->Small printf
   set to 'Yes') calls __io_putchar() */
#define LCD_LOG_PUTCHAR int __io_putchar(int ch)
#else
#define LCD_LOG_PUTCHAR int fputc(int ch, FILE *f)
#endif /* __GNUC__ */


int fputc(int ch, FILE *f)
{    
	static uint16_t x = 0, y = 0;
	
	sFONT *cFont = BSP_LCD_GetFont();
//  int xsize = (BSP_LCD_GetXSize()/cFont->Width);
//	int ysize = (BSP_LCD_GetYSize()/cFont->Height);
	
  if(ch == '\n')    //??
  {
    x = 0;
    y += cFont->Height;
    return ch;
  }

  if(x > BSP_LCD_GetXSize())  
  {
    x = 0;           //x??
    y += cFont->Height;         //y?????
  }
  if(y > BSP_LCD_GetYSize() )  
  {
		y=0;
		BSP_LCD_Clear(LCD_COLOR_WHITE);
  }

	BSP_LCD_DisplayChar(x,y,ch);
  x += cFont->Width ; 

  return ch;
}
