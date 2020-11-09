`include "./bit32_mux4to1.v"
`include "./and_or_fa32.v"

module ALU(a,b,Binvert,Carryin,Operation,Result,CarryOut);
input[31:0] a,b;
input Binvert,Carryin;
input[1:0] Operation;
output[31:0] Result;
output CarryOut;
wire[31:0] o1,o2,o3;
bit32AND m1(o1,a,b);
bit32OR m2(o2,a,b);
FA_dataflow m3(CarryOut,o3,a,({32{Binvert}}&(~b))+((~{32{Binvert}})&b),Binvert);
bit32_mux4to1 m4(Result,Operation,o1,o2,o3,0);
endmodule

module tbALU;
reg Binvert, Carryin;
reg [1:0] Operation;
reg [31:0] a,b;
wire [31:0] Result;
wire CarryOut;
ALU meow(a,b,Binvert,Carryin,Operation,Result,CarryOut);
initial
begin
$monitor($time," a=%32b b=%32b Op=%2b RESULT=%1b %32b",a,b,Operation,CarryOut,Result);
a=32'h0000000B;
b=32'h0000000C;
Operation=2'b00;
Binvert=1'b0;
Carryin=1'b0; //must perform AND resulting in zero
#100 Operation=2'b01; //OR
#100 Operation=2'b10; //ADD
#100 Binvert=1'b1;//SUB
#200 $finish;
end
endmodule