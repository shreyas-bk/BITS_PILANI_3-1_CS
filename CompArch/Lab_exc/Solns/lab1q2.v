module bcd2gray_df(g,b);
input [3:0]b;
output [3:0]g;
assign g[3] = b[3]; //can just make all reg - later find out, is there a difference?
assign g[2] = b[3]^b[2];
assign g[1] = b[2]^b[1];
assign g[0] = b[1]^b[0];
endmodule

module testbench;
reg [3:0]b;
wire [3:0]g;
bcd2gray_df bdf(g,b);
initial
begin
$monitor($time," Input : b=%4b   Output : g = %4b",b,g);
#0 b=4'b0000;
#2 b=4'b0001;
#2 b=4'b0010;
#2 b=4'b0011;
#2 b=4'b0100;
#2 b=4'b0110;
#2 b=4'b0111;
#2 b=4'b1000;
#2 b=4'b1001;
#2 b=4'b0101;
#2 $finish;
end
endmodule