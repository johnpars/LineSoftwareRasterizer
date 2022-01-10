// WARNING: Need to bind the texture and sampler first before dispatches.
Texture2D<float4>   _DebugFont        : register(t0);
SamplerState        _SamplerDebugFont : register(s0);

// -----------------------------------------------------------------------------
// Constants

#define HALF_MAX        65504.0 // (2 - 2^-10) * 2^15
#define HALF_MAX_MINUS1 65472.0 // (2 - 2^-9) * 2^15
#define EPSILON         1.0e-4
#define PI              3.14159265359
#define TWO_PI          6.28318530718
#define FOUR_PI         12.56637061436
#define INV_PI          0.31830988618
#define INV_TWO_PI      0.15915494309
#define INV_FOUR_PI     0.07957747155
#define HALF_PI         1.57079632679
#define INV_HALF_PI     0.636619772367

#define FLT_EPSILON     1.192092896e-07 // Smallest positive number, such that 1.0 + FLT_EPSILON != 1.0
#define FLT_MIN         1.175494351e-38 // Minimum representable positive floating-point number
#define FLT_MAX         3.402823466e+38 // Maximum representable floating-point number

// UX-verified colorblind-optimized debug colors, listed in order of increasing perceived "hotness"
#define DEBUG_COLORS_COUNT 12
#define kDebugColorBlack        float4(0.0   / 255.0, 0.0   / 255.0, 0.0   / 255.0, 1.0) // #000000
#define kDebugColorLightPurple  float4(166.0 / 255.0, 70.0  / 255.0, 242.0 / 255.0, 1.0) // #A646F2
#define kDebugColorDeepBlue     float4(0.0   / 255.0, 26.0  / 255.0, 221.0 / 255.0, 1.0) // #001ADD
#define kDebugColorSkyBlue      float4(65.0  / 255.0, 152.0 / 255.0, 224.0 / 255.0, 1.0) // #4198E0
#define kDebugColorLightBlue    float4(158.0 / 255.0, 228.0 / 255.0, 251.0 / 255.0, 1.0) // #1A1D21
#define kDebugColorTeal         float4(56.0  / 255.0, 243.0 / 255.0, 176.0 / 255.0, 1.0) // #38F3B0
#define kDebugColorBrightGreen  float4(168.0 / 255.0, 238.0 / 255.0, 46.0  / 255.0, 1.0) // #A8EE2E
#define kDebugColorBrightYellow float4(255.0 / 255.0, 253.0 / 255.0, 76.0  / 255.0, 1.0) // #FFFD4C
#define kDebugColorDarkYellow   float4(255.0 / 255.0, 214.0 / 255.0, 0.0   / 255.0, 1.0) // #FFD600
#define kDebugColorOrange       float4(253.0 / 255.0, 152.0 / 255.0, 0.0   / 255.0, 1.0) // #FD9800
#define kDebugColorBrightRed    float4(255.0 / 255.0, 67.0  / 255.0, 51.0  / 255.0, 1.0) // #FF4333
#define kDebugColorDarkRed      float4(132.0 / 255.0, 10.0  / 255.0, 54.0  / 255.0, 1.0) // #840A36

// UX-verified colorblind-optimized "heat color gradient"
static const float4 kDebugColorGradient[DEBUG_COLORS_COUNT] = { kDebugColorBlack, kDebugColorLightPurple, kDebugColorDeepBlue,
    kDebugColorSkyBlue, kDebugColorLightBlue, kDebugColorTeal, kDebugColorBrightGreen, kDebugColorBrightYellow,
    kDebugColorDarkYellow, kDebugColorOrange, kDebugColorBrightRed, kDebugColorDarkRed };

// DebugFont code assume black and white font with texture size 256x128 with bloc of 16x16
#define DEBUG_FONT_TEXT_WIDTH   16
#define DEBUG_FONT_TEXT_HEIGHT  16
#define DEBUG_FONT_TEXT_COUNT_X 16
#define DEBUG_FONT_TEXT_COUNT_Y 8
#define DEBUG_FONT_TEXT_ASCII_START 32

#define DEBUG_FONT_TEXT_SCALE_WIDTH 10 // This control the spacing between characters (if a character fill the text block it will overlap).

bool IsNaN(float x)
{
    return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}

// Using pow often result to a warning like this
// "pow(f, e) will not work for negative f, use abs(f) or conditionally handle negative values if you expect them"
// PositivePow remove this warning when you know the value is positive and avoid inf/NAN.
float PositivePow(float base, float power)
{
    return pow(max(abs(base), float(FLT_EPSILON)), power);
}

float2 PositivePow(float2 base, float2 power)
{
    return pow(max(abs(base), float2(FLT_EPSILON, FLT_EPSILON)), power);
}

float3 PositivePow(float3 base, float3 power)
{
    return pow(max(abs(base), float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON)), power);
}

float4 PositivePow(float4 base, float4 power)
{
    return pow(max(abs(base), float4(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON, FLT_EPSILON)), power);
}

// Only support ASCII symbol from DEBUG_FONT_TEXT_ASCII_START to 126
// return black or white depends if we hit font character or not
// currentUnormCoord is current unormalized screen position
// fixedUnormCoord is the position where we want to draw something, this will be incremented by block font size in provided direction
// color is current screen color
// color of the font to use
// direction is 1 or -1 and indicate fixedUnormCoord block shift
void DrawCharacter(uint asciiValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color, int direction, int fontTextScaleWidth)
{
    // Are we inside a font display block on the screen ?
    uint2 localCharCoord = currentUnormCoord - fixedUnormCoord;
    if (localCharCoord.x >= 0 && localCharCoord.x < DEBUG_FONT_TEXT_WIDTH && localCharCoord.y >= 0 && localCharCoord.y < DEBUG_FONT_TEXT_HEIGHT)
    {
        localCharCoord.y = DEBUG_FONT_TEXT_HEIGHT - localCharCoord.y;

        asciiValue -= DEBUG_FONT_TEXT_ASCII_START; // Our font start at ASCII table 32;
        uint2 asciiCoord = uint2(asciiValue % DEBUG_FONT_TEXT_COUNT_X, asciiValue / DEBUG_FONT_TEXT_COUNT_X);
        // Unorm coordinate inside the font texture
        uint2 unormTexCoord = asciiCoord * uint2(DEBUG_FONT_TEXT_WIDTH, DEBUG_FONT_TEXT_HEIGHT) + localCharCoord;
        // normalized coordinate
        float2 normTexCoord = float2(unormTexCoord) / float2(DEBUG_FONT_TEXT_WIDTH * DEBUG_FONT_TEXT_COUNT_X, DEBUG_FONT_TEXT_HEIGHT * DEBUG_FONT_TEXT_COUNT_Y);

#if UNITY_UV_STARTS_AT_TOP
        normTexCoord.y = 1.0 - normTexCoord.y;
#endif

        float charColor = _DebugFont.SampleLevel(_SamplerDebugFont, normTexCoord, 0.0).r;

        color = color * (1.0 - charColor) + charColor * fontColor;
    }

    fixedUnormCoord.x += fontTextScaleWidth * direction;
}

void DrawCharacter(uint asciiValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color, int direction)
{
    DrawCharacter(asciiValue, fontColor, currentUnormCoord, fixedUnormCoord, color, direction, DEBUG_FONT_TEXT_SCALE_WIDTH);
}

// Shortcut to not have to file direction
void DrawCharacter(uint asciiValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color)
{
    DrawCharacter(asciiValue, fontColor, currentUnormCoord, fixedUnormCoord, color, 1);
}

// Draw a signed integer
// Can't display more than 16 digit
// The two following parameter are for float representation
// leading0 is used when drawing frac part of a float to draw the leading 0 (call is in charge of it)
// forceNegativeSign is used to force to display a negative sign as -0 is not recognize
void DrawInteger(int intValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color, int leading0, bool forceNegativeSign)
{
    const uint maxStringSize = 16;

    uint absIntValue = abs(intValue);

    // 1. Get size of the number of display
    int numEntries = min((intValue == 0 ? 0 : log10(absIntValue)) + ((intValue < 0 || forceNegativeSign) ? 1 : 0) + leading0, maxStringSize);

    // 2. Shift curseur to last location as we will go reverse
    fixedUnormCoord.x += numEntries * DEBUG_FONT_TEXT_SCALE_WIDTH;

    // 3. Display the number
    bool drawCharacter = true; // bit weird, but it is to appease the compiler.
    for (uint j = 0; j < maxStringSize; ++j)
    {
        // Numeric value incurrent font start on the second row at 0
        if(drawCharacter)
            DrawCharacter((absIntValue % 10) + '0', fontColor, currentUnormCoord, fixedUnormCoord, color, -1);

        if (absIntValue  < 10)
            drawCharacter = false;

        absIntValue /= 10;
    }

    // 4. Display leading 0
    if (leading0 > 0)
    {
        for (int i = 0; i < leading0; ++i)
        {
            DrawCharacter('0', fontColor, currentUnormCoord, fixedUnormCoord, color, -1);
        }
    }

    // 5. Display sign
    if (intValue < 0 || forceNegativeSign)
    {
        DrawCharacter('-', fontColor, currentUnormCoord, fixedUnormCoord, color, -1);
    }

    // 6. Reset cursor at end location
    fixedUnormCoord.x += (numEntries + 2) * DEBUG_FONT_TEXT_SCALE_WIDTH;
}

void DrawInteger(int intValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color)
{
    DrawInteger(intValue, fontColor, currentUnormCoord, fixedUnormCoord, color, 0, false);
}

void DrawFloatExplicitPrecision(float floatValue, float3 fontColor, uint2 currentUnormCoord, uint digitCount, inout uint2 fixedUnormCoord, inout float3 color)
{
    if (IsNaN(floatValue))
    {
        DrawCharacter('N', fontColor, currentUnormCoord, fixedUnormCoord, color);
        DrawCharacter('a', fontColor, currentUnormCoord, fixedUnormCoord, color);
        DrawCharacter('N', fontColor, currentUnormCoord, fixedUnormCoord, color);
    }
    else
    {
        int intValue = int(floatValue);
        bool forceNegativeSign = floatValue >= 0.0f ? false : true;
        DrawInteger(intValue, fontColor, currentUnormCoord, fixedUnormCoord, color, 0, forceNegativeSign);
        DrawCharacter('.', fontColor, currentUnormCoord, fixedUnormCoord, color);
        int fracValue = int(frac(abs(floatValue)) * pow(10, digitCount));
        int leading0 = digitCount - (int(log10(fracValue)) + 1); // Counting leading0 to add in front of the float
        DrawInteger(fracValue, fontColor, currentUnormCoord, fixedUnormCoord, color, leading0, false);
    }
}

void DrawFloat(float floatValue, float3 fontColor, uint2 currentUnormCoord, inout uint2 fixedUnormCoord, inout float3 color)
{
    DrawFloatExplicitPrecision(floatValue, fontColor, currentUnormCoord, 6, fixedUnormCoord, color);
}

bool SampleDebugFont(int2 pixCoord, uint digit)
{
    if (pixCoord.x < 0 || pixCoord.y < 0 || pixCoord.x >= 5 || pixCoord.y >= 9 || digit > 9)
        return false;

#define PACK_BITS25(_x0,_x1,_x2,_x3,_x4,_x5,_x6,_x7,_x8,_x9,_x10,_x11,_x12,_x13,_x14,_x15,_x16,_x17,_x18,_x19,_x20,_x21,_x22,_x23,_x24) (_x0|(_x1<<1)|(_x2<<2)|(_x3<<3)|(_x4<<4)|(_x5<<5)|(_x6<<6)|(_x7<<7)|(_x8<<8)|(_x9<<9)|(_x10<<10)|(_x11<<11)|(_x12<<12)|(_x13<<13)|(_x14<<14)|(_x15<<15)|(_x16<<16)|(_x17<<17)|(_x18<<18)|(_x19<<19)|(_x20<<20)|(_x21<<21)|(_x22<<22)|(_x23<<23)|(_x24<<24))
#define _ 0
#define x 1
    uint fontData[9][2] = {
        { PACK_BITS25(_,_,x,_,_,        _,_,x,_,_,      _,x,x,x,_,      x,x,x,x,x,      _,_,_,x,_), PACK_BITS25(x,x,x,x,x,      _,x,x,x,_,      x,x,x,x,x,      _,x,x,x,_,      _,x,x,x,_) },
        { PACK_BITS25(_,x,_,x,_,        _,x,x,_,_,      x,_,_,_,x,      _,_,_,_,x,      _,_,_,x,_), PACK_BITS25(x,_,_,_,_,      x,_,_,_,x,      _,_,_,_,x,      x,_,_,_,x,      x,_,_,_,x) },
        { PACK_BITS25(x,_,_,_,x,        x,_,x,_,_,      x,_,_,_,x,      _,_,_,x,_,      _,_,x,x,_), PACK_BITS25(x,_,_,_,_,      x,_,_,_,_,      _,_,_,x,_,      x,_,_,_,x,      x,_,_,_,x) },
        { PACK_BITS25(x,_,_,_,x,        _,_,x,_,_,      _,_,_,_,x,      _,_,x,_,_,      _,x,_,x,_), PACK_BITS25(x,_,x,x,_,      x,_,_,_,_,      _,_,_,x,_,      x,_,_,_,x,      x,_,_,_,x) },
        { PACK_BITS25(x,_,_,_,x,        _,_,x,_,_,      _,_,_,x,_,      _,x,x,x,_,      _,x,_,x,_), PACK_BITS25(x,x,_,_,x,      x,x,x,x,_,      _,_,x,_,_,      _,x,x,x,_,      _,x,x,x,x) },
        { PACK_BITS25(x,_,_,_,x,        _,_,x,_,_,      _,_,x,_,_,      _,_,_,_,x,      x,_,_,x,_), PACK_BITS25(_,_,_,_,x,      x,_,_,_,x,      _,_,x,_,_,      x,_,_,_,x,      _,_,_,_,x) },
        { PACK_BITS25(x,_,_,_,x,        _,_,x,_,_,      _,x,_,_,_,      _,_,_,_,x,      x,x,x,x,x), PACK_BITS25(_,_,_,_,x,      x,_,_,_,x,      _,x,_,_,_,      x,_,_,_,x,      _,_,_,_,x) },
        { PACK_BITS25(_,x,_,x,_,        _,_,x,_,_,      x,_,_,_,_,      x,_,_,_,x,      _,_,_,x,_), PACK_BITS25(x,_,_,_,x,      x,_,_,_,x,      _,x,_,_,_,      x,_,_,_,x,      x,_,_,_,x) },
        { PACK_BITS25(_,_,x,_,_,        x,x,x,x,x,      x,x,x,x,x,      _,x,x,x,_,      _,_,_,x,_), PACK_BITS25(_,x,x,x,_,      _,x,x,x,_,      _,x,_,_,_,      _,x,x,x,_,      _,x,x,x,_) }
    };
#undef _
#undef x
#undef PACK_BITS25
    return (fontData[8 - pixCoord.y][digit >= 5] >> ((digit % 5) * 5 + pixCoord.x)) & 1;
}

bool SampleDebugFontNumber(int2 pixCoord, uint number)
{
    pixCoord.y -= 4;
    if (number <= 9)
    {
        return SampleDebugFont(pixCoord - int2(6, 0), number);
    }
    else
    {
        return (SampleDebugFont(pixCoord, number / 10) | SampleDebugFont(pixCoord - int2(6, 0), number % 10));
    }
}

// Draws a heatmap with numbered tiles, with increasingly "hot" background colors depending on n,
// where values at or above maxN receive strong red background color.
float4 OverlayHeatMap(uint2 pixCoord, uint2 tileSize, uint n, uint maxN, float opacity)
{
    int colorIndex = 1 + (int)floor(10 * (log2((float)n + 0.1f) / log2(float(maxN))));
    colorIndex = clamp(colorIndex, 0, DEBUG_COLORS_COUNT-1);
    float4 col = kDebugColorGradient[colorIndex];

    int2 coord = (pixCoord & (tileSize - 1)) - int2(tileSize.x/4+1, tileSize.y/3-3);

    float4 color = float4(PositivePow(col.rgb, 2.2), opacity * col.a);
    if (n >= 0)
    {
        if (SampleDebugFontNumber(coord, n))        // Shadow
            color = float4(0, 0, 0, 1);
        if (SampleDebugFontNumber(coord + 1, n))    // Text
            color = float4(1, 1, 1, 1);
    }
    return color;
}

float3 ColorCycle(uint index, uint count)
{
	float t = frac(index / (float)count);

	// Ref: https://www.shadertoy.com/view/4ttfRn
	float3 c = 3.0 * float3(abs(t - 0.5), t.xx) - float3(1.5, 1.0, 2.0);
	return 1.0 - c * c;
}