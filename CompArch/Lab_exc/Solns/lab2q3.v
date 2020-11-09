module FULLADDER(s,c,x,y,z);
input x,y,z;
output reg s,c;
always @(x or y or z)
begin
s = x^y^z;
c = x&y | y&z | z&x;
end
endmodule

module ADDSUB(out,v,a,b,m);
input[3:0] a;
input[3:0] b;
input m;
output v;
output[3:0] out;
wire carry[4:0];
assign carry[0] = m;
generate
	genvar i;
	for(i=0;i<4;i=i+1) begin
		FULLADDER f(out[i],carry[i+1],a[i],m^b[i],carry[i]);
	end
endgenerate
assign v = carry[3]^carry[4];
endmodule

module testbench;
reg[3:0] a;
reg[3:0] b;
reg m;
wire[3:0] out;
wire v;
ADDSUB a1(out,v,a,b,m);
initial
begin
	$monitor($time," a=%4b, b=%4b, m=%1b   sum=%4b v=%1b",a,b,m,out,v);
	#0 a=4'b0000;b=4'b0000;m=1'b0;
	#2 a=4'b0000;b=4'b0001;m=1'b1;
	#2 a=4'b1011;b=4'b1001;m=1'b0;
	#2 a=4'b1000;b=4'b0100;m=1'b1;
	#2 a=4'b0100;b=4'b1100;m=1'b1;
	#5 $finish;
end
endmodule