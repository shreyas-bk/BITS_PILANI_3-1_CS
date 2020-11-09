module decoder(out,x,y,z);
input x,y,z;
output[7:0] out;
wire x0,y0,z0;
not n1(x0,x);
not n2(y0,y);
not n3(z0,z);
and a0(out[0],x0,y0,z0);
and a1(out[1],x0,y0,z);
and a2(out[2],x0,y,z0);
and a3(out[3],x0,y,z);
and a4(out[4],x,y0,z0);
and a5(out[5],x,y0,z);
and a6(out[6],x,y,z0);
and a7(out[7],x,y,z);
endmodule

module fadder(s,c,x,y,z);
input x,y,z;
output s,c;
wire[7:0] out;
decoder d(out,x,y,z);
or o1(s,out[1],out[2],out[4],out[7]);
or o2(c,out[3],out[5],out[6],out[7]);
endmodule

module bit8adder(out,co,a,b,cin);
input[7:0] a;
input[7:0] b;
input cin;
output[7:0] out;
output co;
wire[8:0] carry;
assign carry[0] = cin;
generate // IMPORTANT
	genvar i;
	for(i=0;i<8;i=i+1) begin
		fadder f1(out[i],carry[i+1],a[i],b[i],carry[i]); // can I refernece these outside? If yes, how?
	end	
endgenerate
assign co = carry[8];	
endmodule

module bit32adder(out,co,a,b,cin);
input[31:0] a;
input[31:0] b;
input cin;
output[31:0] out;
output co;
wire[4:0] carry;
assign carry[0] = cin;
generate
	genvar i;
	for(i=0;i<4;i=i+1) begin
		bit8adder f2(out[i*8+7:i*8],carry[i+1],a[i*8+7:i*8],b[i*8+7:i*8],carry[i]);
	end
endgenerate
assign  co = carry[4];
endmodule

module testbench;
 reg[31:0] a;
 reg[31:0] b;
 wire[31:0] out;
 wire co;
 bit32adder fl(out,co,a,b,0);
 initial
 begin
 $monitor(,$time," a=%32b, b=%32b, sum= %1b %32b",a,b,co,out);
 #0 a=32'b01000000000000000000000000000001;b=32'b01000000000000010000000100000001;
 #4 a=32'b01111111011111110111111101111111;b=32'b11000000110000001100000011000000;
 #4 a=32'b01100100011001000110010001100100;b=32'b10000001100000011000000110000001;
 #4 a=32'b00000100000001000000010000000100;b=32'b00000111000001110000011100000111;
 #4 a=32'b00001000000010000000100000001000;b=32'b10000000100000001000000010000000;
 end
endmodule



